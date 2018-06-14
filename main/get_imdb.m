function imdb = get_imdb(imdb, opts, net)
if ~isempty(strfind(opts.modelType, 'fc')) || opts.imageSize <= 0
    if strcmp(opts.dataset, 'labelme')
        imdbName = sprintf('%s_gist', opts.dataset);
    else
        imdbName = sprintf('%s_fc7', opts.dataset);
    end
else
    imdbName = opts.dataset;
end
imdbFunc = str2func(['imdb.' imdbName]);

% normalize images/features?
if ismember(opts.imageSize, [224 227])
    assert(~opts.normalize);
end
if opts.normalize
    imdbName = [imdbName '_normalized'];
end
if ismember(opts.dataset, {'cifar', 'nus'})
    imdbName = sprintf('%s_split%d', imdbName, opts.split);
end
if strcmp(opts.dataset, 'imagenet')
    imdbName = sprintf('%s_%dx%d', imdbName, opts.imageSize, opts.imageSize);
end
% imdbName finalized
if ~isempty(imdb) && strcmp(imdb.name, imdbName)
    myLogInfo('%s already loaded', imdb.name);
    return;
end
imdb = [];

% imdbFile
imdbFile = fullfile(opts.dataDir, ['imdb_' imdbName '.mat']);
myLogInfo(imdbFile);

% load/save
t0 = tic;
if exist(imdbFile, 'file')
    imdb = load(imdbFile) ;
    myLogInfo('loaded in %.2fs', toc(t0));
else
    imdb = imdbFunc(opts, net) ;
    save(imdbFile, '-struct', 'imdb', '-v7.3') ;
    if ~opts.windows, unix(['chmod g+rw ' imdbFile]); end
    myLogInfo('saved in %.2fs', toc(t0));
end
imdb.name = imdbName;
myLogInfo('%s loaded', imdb.name);

end
