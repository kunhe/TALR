function test_unsupervised(net, imdb, batchFunc, opts, metrics, ...
        noLossLayer, subset)
assert(~isempty(metrics));
if ~iscell(metrics)
    assert(isstr(metrics));
    metrics = {metrics};
end
if ~exist('noLossLayer', 'var')
    noLossLayer = false;
end
if noLossLayer,  layerOffset = 0;
else, layerOffset = 1;
end
assert(opts.unsupervised);

train_id = find(imdb.images.set == 1 | imdb.images.set == 2);
test_id  = find(imdb.images.set == 3);
if exist('subset', 'var')
    train_id = train_id(randperm(numel(train_id), subset(2)));
    test_id = test_id(randperm(numel(test_id), subset(1)));
end
Ntrain = numel(train_id);
Ntest  = numel(test_id);

batch_size = opts.batchSize;
onGPU = ~isempty(opts.gpus);

% get Htest and Xtest
fprintf('Getting (Htest, Xtest)...'); tic;
Htest = zeros(opts.nbits, Ntest, 'single');
Xtest = [];
for t = 1:batch_size:Ntest
    ed = min(t+batch_size-1, Ntest);
    [rex, data] = cnn_encode1(net, batchFunc, imdb, test_id(t:ed), ...
        onGPU, layerOffset);
    Htest(:, t:ed) = single(rex > 0);
    Xtest(t:ed, :) = squeeze(data)';
end
toc;

% get Htrain and fill in Aff incrementally
fprintf('Getting (Htrain, Aff)...'); tic;
Htrain = zeros(opts.nbits, Ntrain, 'single');
Aff_bin = zeros(Ntest, Ntrain, 'logical');
Aff_mlv = zeros(Ntest, Ntrain, 'int8');
for t = 1:batch_size:Ntrain
    ed = min(t+batch_size-1, Ntrain);
    [rex, data] = cnn_encode1(net, batchFunc, imdb, train_id(t:ed), ...
        onGPU, layerOffset);
    data = squeeze(data)';
    Htrain(:, t:ed)  = single(rex > 0);
    Aff_bin(:, t:ed) = affinity_binary([], [], Xtest, data, opts);
    Aff_mlv(:, t:ed) = affinity_multlv([], [], Xtest, data, opts);
end
toc;

% evaluate
fprintf('\n');
myLogInfo('[%d bits] Evaluating %d queries ...', opts.nbits, numel(test_id));
for m = metrics
    if ~isempty(strfind(m{1}, 'AP'))
        Aff = Aff_bin;
    else
        Aff = Aff_mlv;
    end
    if ~isempty(strfind(m{1}, '@'))
        s = strsplit(m{1}, '@');
        assert(numel(s) == 2);
        cutoff = str2num(s{2});
        evalFn = str2func(['eval.' s{1}]);
    else
        cutoff = [];
        evalFn = str2func(['eval.' m{1}]);
    end
    evalFn(Htest, Htrain, Aff, opts);
end
fprintf('\n');
end

% ----------------------------------------------------------------------
function [rex, data] = cnn_encode1(net, batchFunc, imdb, ids, onGPU, layerOffset)
[data, labels] = batchFunc(imdb, ids);
net.layers{end}.class = labels;
if onGPU
    data = gpuArray(data); 
    res = vl_simplenn(net, data, [], [], 'mode', 'test');
    rex = squeeze(gather(res(end-layerOffset).x));
else
    res = vl_simplenn(net, data, [], [], 'mode', 'test');
    rex = squeeze(res(end-layerOffset).x);
end
end
