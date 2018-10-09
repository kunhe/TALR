function demo_NDCG(dataset, nbits, modelType, varargin)

% init opts
ip = inputParser;
ip.addParameter('split', 1);
ip.addParameter('obj', 'ndcgr_s');
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = get_opts(ip.Results, dataset, nbits, modelType, varargin{:});

%%%%%%%%%%%%%%%%% hard-coded fields %%%%%%%%%%%%%%%%%
% Evaluation Metrics
% tieNDCG: tie-aware NDCG, evaluated on the full ranking
% NDCG@k : tie-agnostic NDCG, evaluated at cutoff k
%
opts.metrics = {'tieNDCG'};
if strcmp(dataset, 'labelme')
    opts.metrics{end+1} = 'tieAP';
    opts.modelType = 'fc1';
end
%%%%%%%%%%%%%%%%% hard-coded fields %%%%%%%%%%%%%%%%%

% post-parsing
cleanupObj = onCleanup(@cleanup);
opts = process_opts(opts);  % carry out all post-processing on opts
record_diary(opts);
opts
myLogInfo(opts.methodID);
myLogInfo(opts.identifier);

% ---------------
% model & data
% ---------------
if ~isempty(opts.gpus) && opts.gpus == 0
    opts.gpus = auto_select_gpu;
end
[net, opts] = get_model(opts);

global IMDB
IMDB = get_imdb(IMDB, opts, net);
disp(IMDB.images)
if isempty(IMDB.images.labels)
    assert(opts.unsupervised);
    itrain = find(IMDB.images.set == 1);
    Xtrain = squeeze(IMDB.images.data(:, :, :, itrain))';
    opts.thr_dist = prctile(pdist(Xtrain), [.1 .2 1 5]);
    opts.Aff = affinity_multlv([], [], Xtrain, Xtrain, opts);
    net.layers{end}.opts = opts;
    opts.thr_dist
end

% ---------------
% train
% ---------------
if ismember(opts.modelType, {'fc', 'fc1'})
    batchFunc = @batch_fc7;
elseif strcmp(opts.modelType, 'nin')
    % NIN on orig images
    batchFunc = @batch_simplenn;
else
    % imagenet model
    imgSize = opts.imageSize;
    meanImage = single(net.meta.normalization.averageImage);
    if isequal(size(meanImage), [1 1 3])
        meanImage = repmat(meanImage, [imgSize imgSize]);
    else
        assert(isequal(size(meanImage), [imgSize imgSize 3]));
    end
    batchFunc = @(I, B) batch_imagenet(I, B, imgSize, meanImage);
end

% figure out learning rate vector
if opts.lrdecay>0 & opts.lrdecay<1
    % constant decay
    assert(opts.lrstep > 0);
    cur_lr = opts.lr;
    lrvec = [];
    while length(lrvec) < opts.epochs
        lrvec = [lrvec, ones(1, opts.lrstep)*cur_lr];
        cur_lr = cur_lr * opts.lrdecay;
    end
    save_eps = [0 : opts.lrstep : opts.epochs];
elseif opts.lrdecay >= 1
    % linear decay
    assert(mod(opts.lrdecay, 1) == 0);
    lrvec = linspace(opts.lr, 0, opts.lrdecay+1);
    opts.epochs = min(opts.epochs, opts.lrdecay);
    save_eps = [opts.epochs];
else
    % no decay
    lrvec = opts.lr;
    save_eps = [opts.epochs];
end

[net, info] = train_simplenn(net, IMDB, batchFunc, ...
    'continue'       , opts.continue      , ...
    'debug'          , opts.debug         , ...
    'expDir'         , opts.expDir        , ...
    'batchSize'      , opts.batchSize     , ...
    'numEpochs'      , opts.epochs        , ...
    'saveEpochs'     , [opts.epochs]      , ...
    'learningRate'   , lrvec              , ...
    'weightDecay'    , opts.wdecay        , ...
    'val'            , NaN                , ...
    'gpus'           , opts.gpus          , ...
    'errorFunction'  , 'none'             , ...
    'epochCallback'  , @epoch_callback) ;

% ---------------
% test
% ---------------
if mod(opts.epochs, opts.testInterval)
    opts.testFunc(net, IMDB, batchFunc, opts, opts.metrics);
end
net = vl_simplenn_move(net, 'cpu'); 
diary('off');
end


% -------------------------------------------------------------------
% postprocessing after each epoch
% -------------------------------------------------------------------
function net = epoch_callback(epoch, net, IMDB, batchFunc, netopts)
opts = net.layers{end}.opts;
% disp
myLogInfo('[%s]', opts.methodID);
myLogInfo('[%s]', opts.identifier);
myLogInfo('%s', char(datetime));
if ~isempty(opts.gpus)
    [~, name] = unix('hostname');
    myLogInfo('GPU #%d on %s', opts.gpus, name);
end
% test?
if ~isfield(opts, 'testFunc'), opts.testFunc = @test_supervised; end
if ~isfield(opts, 'testInterval'), opts.testInterval = 10; end
if ~isfield(opts, 'metrics'), opts.metics = {'tieNDCG'}; end
if ~mod(epoch, opts.testInterval)
    opts.testFunc(net, IMDB, batchFunc, opts, opts.metrics);
end
% slope annealing trick
if numel(opts.gamma)>1 && opts.gamma(2)>1 && ~mod(epoch,opts.gamma(2))
    % anneal rate: default 2, or override by gamma(3)
    gamma_prev = opts.gamma_p;
    if numel(opts.gamma) >= 3
        opts.gamma_p = opts.gamma(3) * opts.gamma_p;
    else
        opts.gamma_p = 2 * opts.gamma_p;
    end
    net.layers{end}.opts = opts;
    myLogInfo('\gamma annealing: %g -> %g', gamma_prev, opts.gamma_p);
end
diary off, diary on
end
