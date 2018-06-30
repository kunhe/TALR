function demo_AP(dataset, nbits, modelType, varargin)
% AP experiments on CIFAR-10 and NUS-WIDE

% init opts
ip = inputParser;
ip.addParameter('split', 1);  % exp. setting, current options: {1, 2}
ip.addParameter('obj', 'apr_s');  % optimization obj, current options: {'apr_s'}
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = get_opts(ip.Results, dataset, nbits, modelType, varargin{:});

%%%%%%%%%%%%%%%%% hard-coded fields %%%%%%%%%%%%%%%%%
% Evaluation Metrics
% tieAP: tie-aware AP, evaluated on the full ranking
% AP@k : tie-agnostic AP, evaluated at cutoff k
%
% NOTE only the tie-agnostic AP requires installation of VLFeat
% TODO remove VLFeat dependency in the future
%
opts.metrics = {'tieAP'};
if strcmp(opts.dataset, 'cifar')
    opts.metrics{end+1} = 'AP';
elseif strcmp(opts.dataset, 'nus')
    opts.metrics{end+1} = 'AP@5000';
    opts.metrics{end+1} = 'AP@50000';
elseif strcmp(opts.dataset, 'imagenet')
    opts.metrics{end+1} = 'AP@1000';
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

% IMDB
global IMDB
IMDB = get_imdb(IMDB, opts, net);
disp(IMDB.images)

% ---------------
% train
% ---------------
if strcmp(opts.dataset, 'imagenet')
    % finetune on ImageNet
    batchFunc = @imdb.batch_simplenn;
else
    % finetune ImageNet model on CIFAR-10 or NUS-WIDE
    imgSize = opts.imageSize;
    meanImage = single(net.meta.normalization.averageImage);
    if isequal(size(meanImage), [1 1 3])
        meanImage = repmat(meanImage, [imgSize imgSize]);
    else
        assert(isequal(size(meanImage), [imgSize imgSize 3]));
    end
    batchFunc = @(I, B) imdb.batch_imagenet(I, B, imgSize, meanImage);
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
    'saveEpochs'     , save_eps           , ...
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
if ~isfield(opts, 'metrics'), opts.metics = {'tieAP', 'AP'}; end
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
diary off, diary on  % flush diary
end
