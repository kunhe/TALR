function Aff = affinity_mltlv(Y1, Y2, X1, X2, opts)
% multi-level affinity, returns int8 matrix
if opts.unsupervised
    assert(~isempty(X1) && ~isempty(X2));
    [N1, Dx]  = size(X1);
    [N2, Dx2] = size(X2); assert(Dx2 == Dx);
    Aff = zeros(N1, N2, 'int8');
    raw_dist = bsxfun(@plus, sum(X1.^2, 2), sum(X2.^2, 2)') - 2*X1*X2';
    % if there are multiple thresholds: 
    % smaller the threshold, larger the affinity (integer-valued)
    thrs = sort(opts.thr_dist, 'descend');
    affs = [1 2 5 10];
    for k = 1:length(thrs)
        Aff(raw_dist <= thrs(k)) = affs(k);
    end
else
    assert(~isempty(Y1) && ~isempty(Y2));
    [N1, Dy] = size(Y1);
    [N2, Dy2] = size(Y2); assert(Dy2 == Dy);
    if Dy == 1
        Aff = bsxfun(@eq, Y1, Y2');
    else
        Aff = Y1 * Y2';
    end
    Aff = int8(Aff);
end

end
