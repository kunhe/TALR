function H = cnn_encode(net, batchFunc, imdb, ids, opts, ...
    noLossLayer, verbose)
if ~exist('verbose', 'var'), verbose = true; end
if ~noLossLayer
    net.layers(end) = [];
end
batch_size = opts.batchSize;
onGPU = ~isempty(opts.gpus);

if verbose, 
    myLogInfo('Testing [%s] on %d images ...', opts.modelType, length(ids)); 
    t0 = tic; tic;
end
H = zeros(opts.nbits, length(ids), 'single');
for t = 1:batch_size:length(ids)
    ed = min(t+batch_size-1, length(ids));
    [data, labels] = batchFunc(imdb, ids(t:ed));
    net.layers{end}.class = labels;
    if onGPU
        data = gpuArray(data); 
        res = vl_simplenn(net, data, [], [], 'mode', 'test', ...
            'conserveMemory', true, 'cudnn', true);
        rex = squeeze(gather(res(end).x));
    else
        res = vl_simplenn(net, data, [], [], 'mode', 'test');
        rex = squeeze(res(end).x);
    end
    H(:, t:ed) = single(rex > 0);
    if verbose && toc > 100
        myLogInfo('%6d / %d', t, length(ids)); 
        tic;
    end
end
if verbose, toc(t0); end
end
