function [net, opts] = get_model(opts)
t0 = tic;
modelFunc = str2func(sprintf('models.%s', opts.modelType));
[net, opts.imageSize, opts.normalize] = modelFunc(opts);
myLogInfo('%s in %.2fs', opts.modelType, toc(t0));
end
