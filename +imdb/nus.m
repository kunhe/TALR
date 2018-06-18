function DB = nus(opts, net)

sdir = fullfile(opts.dataDir, 'NUSWIDE_images');

images = textread([sdir '/Imagelist.txt'], '%s');
images = strrep(images, '\', '/');
images = strrep(images, 'C:/ImageData', sdir);

% get labels
labels = load([sdir '/AllLabels81.txt']);
myLogInfo('Total images = %g', size(labels, 1));

% use 21 most frequent labels only
myLogInfo('Keeping 21 most frequent tags, removing rest...');
[val, sel] = sort(sum(labels, 1), 'descend');
labels = labels(:, sel(1:21));
myLogInfo('Min tag freq %d', val(21));

% remove those without any labels
keep   = sum(labels, 2) > 0;
labels = labels(keep, :);
images = images(keep);
assert(size(labels, 1) == length(images));
myLogInfo('Keeping # images = %g', sum(keep));

% split
sets = imdb.split_nus(labels, opts);

% save
DB.images.data = images;  % only save image names, load on demand
DB.images.labels = single(labels)';
DB.images.set = uint8(sets)';
DB.meta.sets = {'train', 'val', 'test'} ;
end
