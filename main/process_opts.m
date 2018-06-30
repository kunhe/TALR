function opts = process_opts(opts)
% post-parse processing

% dataset
if ismember(opts.dataset, {'cifar', 'nus', 'imagenet'})
    opts.unsupervised = false;
    opts.testFunc = @test_supervised;
% TODO elseif strcmp(opts.dataset, 'labelme')
%    opts.testFunc = @test_unsupervised;
%    opts.unsupervised = true;
else
    error('unknown dataset');
end

% tanh approx
opts.gamma_p = opts.gamma(1);
if numel(opts.gamma) < 3
    % default annealing factor: 2x
    opts.gamma(3) = 2; 
end

% methodID
opts.methodID = upper(opts.dataset);
if ismember(opts.dataset, {'cifar'})
    opts.methodID = sprintf('%s-S%d', opts.methodID, opts.split);
end
opts.methodID = sprintf('%s-%dbits-%s-%s', opts.methodID, opts.nbits, ...
    opts.modelType, upper(opts.obj));

% identifier
% 0. shared
idr = sprintf('Bin%d', opts.nbins);
if opts.delta ~= 1
    idr = sprintf('%sDelta%g', idr, opts.delta);
end
if opts.gamma(2) > 0
    idr = sprintf('%s-Gamma%gx%ge%g-batch%d-LR%gD%g', idr, ...
        opts.gamma(1), opts.gamma(3), opts.gamma(2), opts.batchSize, ...
        opts.lr, opts.lrdecay);
else
    idr = sprintf('%s-Gamma%g-batch%d-LR%gD%g', idr, ...
        opts.gamma(1), opts.batchSize, ...
        opts.lr, opts.lrdecay);
end
% 1. lr decay
if opts.lrdecay > 0
    idr = sprintf('%sE%d', idr, opts.lrstep);
end
% 2. weight decay
idr = sprintf('%s-wd%g', idr, opts.wdecay);
% 3. dropout
if opts.dropout > 0 && opts.dropout < 1
    idr = sprintf('%s-Dropout%g', idr, opts.dropout);
end
% 3. lr multiplier for pretrained layers
if ~strcmp(opts.modelType, 'fc1')
    if opts.lrmult > 1 || opts.lrmult < 0
        opts.lrmult = 0.1;  % default to 0.1
        myLogInfo('Warning: opts.lrmult outside [0, 1]');
        myLogInfo('         opts.lrmult <- %g', opts.lrmult);
    end
    idr = sprintf('%s-lrmult%g', idr, opts.lrmult);
end
opts.identifier = idr;

% --------------------------------------------
% generic
opts.localDir = './cachedir';  % use symlink on linux
if ~exist(opts.localDir, 'file')
    error('Please make a symlink for cachedir!');
end
opts.dataDir = './data';

% --------------------------------------------
% expDir: format like .../deep-hashing/deepMI-cifar32-fc
opts.expDir = fullfile(opts.localDir, opts.methodID);
if exist(opts.expDir, 'dir') == 0, 
    mkdir(opts.expDir);
    unix(['chmod g+rw ' opts.expDir]);
end

% --------------------------------------------
% identifier string for the current experiment
idr = opts.identifier;
if isempty(opts.prefix) % default prefix: timestamp
    opts.prefix = yyyymmdd(datetime('now'));
end
opts.identifier = [opts.prefix '-' idr];

% --------------------------------------------
% expand expDir
opts.expDir = fullfile(opts.expDir, opts.identifier);
if ~exist(opts.expDir, 'dir'),
    myLogInfo(['creating opts.expDir: ' opts.expDir]);
    mkdir(opts.expDir);
    unix(['chmod g+rw ' opts.expDir]);
end

end
