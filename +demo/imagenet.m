function demo_imagenet(nbits, varargin)

% init opts
ip = inputParser;
ip.addParameter('modelType', 'alexnet');
ip.addParameter('obj'      , 'apr_s');
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts = get_opts(opts, 'imagenet', nbits, opts.modelType, varargin{:});

%%%%%%%%%%%%%%%%% hard-coded fields %%%%%%%%%%%%%%%%%
opts.metrics = {'tieAP', 'AP@1000'};
opts.testFunc = @test_imagenet;
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
IMDB.images.set(1:numel(IMDB.images.labels)) = 1;
disp(IMDB.images)

% ---------------
% train
% ---------------
sz = [opts.imageSize opts.imageSize];
meanImage = single(net.meta.normalization.averageImage);
if isequal(size(meanImage), [1 1 3])
    meanImage = repmat(meanImage, sz);
else
    assert(isequal(size(meanImage), [sz 3]));
end
testBatchFunc = @(I, B) imdb.batch_imagenet(I, B, opts.imageSize, meanImage);
trainBatchFunc = @imdb.batch_simplenn;

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

[net, info] = train_imagenet(net, IMDB, trainBatchFunc, testBatchFunc, ...
    'continue'       , opts.continue      , ...
    'debug'          , opts.debug         , ...
    'plotStatistics' , opts.plot          , ...
    'expDir'         , opts.expDir        , ...
    'batchSize'      , opts.batchSize     , ...
    'numEpochs'      , opts.epochs        , ...
    'saveEpochs'     , [opts.epochs]      , ...
    'learningRate'   , lrvec              , ...
    'weightDecay'    , opts.wdecay        , ...
    'backPropDepth'  , opts.bpdepth       , ...
    'val'            , NaN                , ...
    'gpus'           , opts.gpus          , ...
    'errorFunction'  , 'none'             , ...
    'epochCallback'  , @epoch_callback) ;

net = vl_simplenn_move(net, 'cpu'); 
diary('off');
end


% -------------------------------------------------------------------
% postprocessing after each epoch
% -------------------------------------------------------------------
function net = epoch_callback(epoch, net, IMDB, testBatchFunc)
opts = net.layers{end}.opts;
% disp
myLogInfo('[%s]', opts.methodID);
myLogInfo('[%s]', opts.identifier);
myLogInfo('%s', char(datetime));
if ~isempty(opts.gpus)
    [~, name] = unix('hostname');
    myLogInfo('GPU #%d on %s', opts.gpus, name);
end
% test
imdb_test = [];
imdb_test.images = IMDB.images.all;
if ~mod(epoch, opts.testInterval) || epoch == opts.epochs
    test_imagenet(net, imdb_test, testBatchFunc, opts, opts.metrics);
end
% slope annealing
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
