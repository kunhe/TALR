function test_supervised(net, imdb, batchFunc, opts, metrics, ...
    noLossLayer, subset)

assert(~isempty(metrics));
if ~iscell(metrics)
    assert(isstr(metrics));
    metrics = {metrics};
end
if ~exist('noLossLayer', 'var')
    noLossLayer = false;
end
opts.unsupervised = false;

train_id = find(imdb.images.set == 1 | imdb.images.set == 2);
test_id  = find(imdb.images.set == 3);
if exist('subset', 'var')
    myLogInfo('Sampling random subset: %d test, %d database', subset(1), subset(2));
    test_id = test_id(randperm(numel(test_id), subset(1)));
    train_id = train_id(randperm(numel(train_id), subset(2)));
end
Ytrain = imdb.images.labels(:, train_id)';
Ytest  = imdb.images.labels(:, test_id)';

% hash codes
Htest  = cnn_encode(net, batchFunc, imdb, test_id , opts, noLossLayer);
Htrain = cnn_encode(net, batchFunc, imdb, train_id, opts, noLossLayer);

% evaluate
fprintf('\n');
myLogInfo('[%d bits] Evaluating %d queries ...', opts.nbits, numel(test_id));
for m = metrics
    % available metics: tieAP, tieNDCG, AP, AP@N, NDCG, NDCG@N
    if ~isempty(strfind(m{1}, 'AP'))
        Aff = affinity_binary(Ytest, Ytrain, [], [], opts);
    else
        Aff = affinity_multlv(Ytest, Ytrain, [], [], opts);
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
    evalFn(Htest, Htrain, Aff, opts, cutoff);
end
fprintf('\n');
end
