function [fc7s, labels] = batch_fc7(imdb, batch)
fc7s = imdb.images.data(:, :, :, batch) ;
if isempty(imdb.images.labels)
    % for unsupervised
    itrain = find(imdb.images.set == 1);
    [~, labels] = ismember(batch, itrain);
else
    labels = permute(imdb.images.labels(:, batch), [3 4 2 1]);
end
end
