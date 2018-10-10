function res = tieNDCG(Htest, Htrain, Aff, opts, cutoff, varargin)
% input: 
%   Htrain - (logical) training binary codes
%   Htest  - (logical) testing binary codes

[nbits, Ntrain] = size(Htrain);
assert(size(Htest, 1) == nbits);
Ntest = size(Htest, 2);

Discount = 1 ./ log2((1:Ntrain) + 1);
CD = [0, cumsum(Discount)];

t0 = tic;
Aff = int8(Aff);
Aff_sorted = sort(Aff, 2, 'Descend');
DCGi = single(2.^Aff_sorted-1) * Discount';

DCGt = zeros(Ntest, 1);
for i = 1:Ntest
    hdist = (nbits - (2*Htest(:, i)-1)'*(2*Htrain-1))/2;
    nd = accumarray(hdist'+1, 1, [nbits+1, 1]);
    Nd = cumsum(nd);
    nz = find(nd);

    Ghat = accumarray(hdist'+1, 2.^Aff(i, :)-1, [nbits+1, 1]);
    Dbar = CD(Nd(nz)+1) - CD(Nd(nz)-nd(nz)+1);
    DCGt(i) = dot(Ghat(nz)./nd(nz), Dbar);
end

NDCGt = DCGt ./ DCGi;
NDCGt(DCGi == 0) = 1;
myLogInfo('NDCG = %g (tie-aware)', mean(NDCGt));
toc(t0);
res = mean(NDCGt);
end
