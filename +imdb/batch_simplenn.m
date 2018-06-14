function [images, labels] = getSimpleNNBatch(imdb, batch)
images = imdb.images.data(:, :, :, batch) ;
if rand > 0.5, images=fliplr(images) ; end
if isempty(imdb.images.labels)
    labels = [];
else
    labels = permute(imdb.images.labels(:, batch), [3 4 2 1]);
end
end
