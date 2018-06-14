function Aff = affinity_binary(Y1, Y2, X1, X2, opts)
% binary affinity, returns logical matrix
if opts.unsupervised
    assert(~isempty(X1) && ~isempty(X2));
    [N1, Dx]  = size(X1);
    [N2, Dx2] = size(X2); assert(Dx2 == Dx);
    raw_dist = bsxfun(@plus, sum(X1.^2, 2), sum(X2.^2, 2)') - 2*X1*X2';
    Aff = (raw_dist <= max(opts.thr_dist));
else
    assert(~isempty(Y1) && ~isempty(Y2));
    [N1, Dy] = size(Y1);
    [N2, Dy2] = size(Y2); assert(Dy2 == Dy);
    if Dy == 1
        Aff = bsxfun(@eq, Y1, Y2');
    else
        Aff = (Y1 * Y2' > 0);
    end
end

end
