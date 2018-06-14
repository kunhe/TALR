function [images, labels] = batch_imagenet(imdb, batch, imgSize, meanImage)

% get images
if ~iscell(imdb.images.data)
    % already loaded in imdb
    images = imdb.images.data(:, :, :, batch) ;
    % normalization
    if imgSize ~= size(images, 1)
        images = imresize(images, [imgSize, imgSize]);
    end
    images = bsxfun(@minus, images, meanImage);
    % get labels
    if isempty(imdb.images.labels)
        itrain = find(imdb.images.set == 1);
        [~, labels] = ismember(batch, itrain);
    else
        labels = permute(imdb.images.labels(:, batch), [3 4 2 1]);
    end
else
    args = {'Gpu', 'Pack', ...
            'NumThreads', 4, ...
            'Resize', [imgSize imgSize], ...
            'Interpolation', 'bicubic', ...
            'subtractAverage', meanImage};
    % train or test? train: use data augmentation
    if imdb.images.set(batch(1)) == 1
        args{end+1} = 'Flip';
    end
    % imdb.images.data is a cell array of filepaths
    % first call: prefetch
    vl_imreadjpeg(imdb.images.data(batch), args{:}, 'prefetch');
    % get labels now
    if isempty(imdb.images.labels)
        itrain = find(imdb.images.set == 1);
        [~, labels] = ismember(batch, itrain);
    else
        labels = permute(imdb.images.labels(:, batch), [3 4 2 1]);
    end
    % second call to actually get images
    images = vl_imreadjpeg(imdb.images.data(batch), args{:});
    images = images{1};
end

end
