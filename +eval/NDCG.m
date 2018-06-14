function res = NDCG(Htest, Htrain, Aff, opts, cutoff, bit_weights)
% input: 
%   Htrain - (logical) training binary codes
%   Htest  - (logical) testing binary codes
%   Aff    - Ntest x Ntrain affinity matrix

[nbits, Ntest] = size(Htest);
if isfield(opts, 'nbits'), assert(nbits == opts.nbits); end
assert(size(Htrain, 1) == nbits);
Ntrain = size(Htrain, 2);
if isempty(cutoff)
    cutoff = Ntrain;
    %myLogInfo('Eval full ranking');
    NDCGname = 'NDCG';
else
    cutoff = min(cutoff, Ntrain);
    myLogInfo('Ranking cutoff = %d', cutoff);
    NDCGname = sprintf('NDCG@%d', cutoff);
end

phi_t = 2*Htest  - 1;
phi_r = 2*Htrain - 1; 
hdist = (nbits - phi_t' * phi_r)/2;  % pairwise dist matrix
Aff   = max(0, Aff);
disp([0, fliplr(opts.thr_dist); single(unique(Aff(:))')])

DCGi = zeros(Ntest, 1);
DCGr = DCGi;
DCGo = DCGi;
DCGp = DCGi;
discount = 1 ./ log2((1:Ntrain) + 1);
t0 = tic;
for i = 1 : Ntest
    if all(Aff(i, :) == 0)
        DCGi(i) = 1; DCGr(i) = 1; DCGo(i) = 1; DCGp(i) = 1;
        continue;
    end
    G = 2.^(Aff(i, :)) - 1;
    D = hdist(i, :);

    % ideal DCG
    [Gi, Io] = sort(G, 'descend');  % ideal ordering
    DCGi(i) = dot(single(Gi(1:cutoff)), discount(1:cutoff));

    Ip = fliplr(Io);
    Do = D(Io);
    Dp = D(Ip);
    ir = [];  % regular (tie-unaware)
    io = [];  % optimistic
    ip = [];  % pessimistic
    n  = 0;
    for d = 0 : nbits
        id = find(D == d);
        ir = [ir, id];
        io = [io, Io(Do == d)];
        ip = [ip, Ip(Dp == d)];
        n  = n + length(id);
        if n >= cutoff, break; end
    end
    G = single(G);
    DCGr(i) = dot(G(ir(1:cutoff)), discount(1:cutoff));
    DCGo(i) = dot(G(io(1:cutoff)), discount(1:cutoff));
    DCGp(i) = dot(G(ip(1:cutoff)), discount(1:cutoff));
end

DCGr = DCGr ./ DCGi;
DCGo = DCGo ./ DCGi;
DCGp = DCGp ./ DCGi;

myLogInfo('%s == %g (random tiebreak)', NDCGname, mean(DCGr));
myLogInfo('%s <= %g (upper bound)', NDCGname, mean(DCGo));
myLogInfo('%s >= %g (lower bound)', NDCGname, mean(DCGp));
toc(t0);
res = [mean(DCGr), mean(DCGo), mean(DCGp)];
end
