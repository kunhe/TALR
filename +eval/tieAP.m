function res = tieAP(Htest, Htrain, Aff, opts, cutoff, varargin)
% input: 
%   Htrain - (logical) training binary codes
%   Htest  - (logical) testing binary codes
%   Aff    - Ntest x Ntrain affinity matrix

[nbits, Ntest] = size(Htest);
if isfield(opts, 'nbits'), assert(nbits == opts.nbits); end
assert(size(Htrain, 1) == nbits);
Ntrain = size(Htrain, 2);

phi_t = 2*Htest  - 1;
phi_r = 2*Htrain - 1; 
hdist = (nbits - phi_t' * phi_r)/2;  % pairwise dist matrix
Aff   = (Aff > 0);
Np    = sum(Aff, 2);

t0 = tic;
APt = zeros(1, Ntest);
for i = 1:Ntest
    if Np(i) == 0, continue; end

    nD  = accumarray(hdist(i, :)'+1, 1, [nbits+1, 1]);
    nDp = accumarray(hdist(i, Aff(i, :))'+1, 1, [nbits+1, 1]);
    NDp = cumsum(nDp);
    ND  = cumsum(nD);

    Np0 = zeros(size(NDp));
    N0  = zeros(size(ND ));
    Np0(2:end) = NDp(1:end-1);
    N0 (2:end) = ND (1:end-1);

    % 1. exact solution for no ties
    APt_i = nDp .* NDp ./ ND;
    APt_i(ND == 0) = 0;

    % 2. update where there are ties
    for l = find(nD>1 & nDp>0)'
        mult = (nDp(l) - 1) / (nD(l) - 1);
        nume = Np0(l) + 1 + (0:nD(l)-1) * mult;
        deno = N0(l) + (1:nD(l));
        APt_i(l) = mean(nume./deno) * nDp(l);
    end
    AP_t(i) = sum(APt_i) / Np(i);
end
myLogInfo('AP = %g (tie-aware)', mean(AP_t));
toc(t0);
res = mean(AP_t);
end
