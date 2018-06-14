function res = AP(Htest, Htrain, Aff, opts, cutoff, bit_weights)
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
    APname = 'AP';
else
    cutoff = min(cutoff, Ntrain);
    myLogInfo('Ranking cutoff = %d', cutoff);
    APname = sprintf('AP@%d', cutoff);
end

t0 = tic;
Aff   = (Aff > 0);
phi_t = 2*Htest  - 1;
phi_r = 2*Htrain - 1; 
hdist = (nbits - phi_t' * phi_r)/2;  % pairwise dist matrix

AP = zeros(Ntest, 1);
APUB = AP;
APLB = AP;
for i = 1:Ntest
    A  = Aff(i, :);
    D  = hdist(i, :);
    ir = [];  % regular (tie-unaware)
    io = [];  % optimistic
    ip = [];  % pessimistic
    n  = 0;
    for d = 0 : nbits
        id = find(D == d);
        ir = [ir, id];
        io = [io, id( A(id)), id(~A(id))];
        ip = [ip, id(~A(id)), id( A(id))];
        n  = n + length(id);
        if n >= cutoff, break; end
    end
    AP(i) = get_AP( A(ir(1:cutoff)) );
    APUB(i) = get_AP( A(io(1:cutoff)) );
    APLB(i) = get_AP( A(ip(1:cutoff)) );
end

myLogInfo('%s == %g (random tiebreak)', APname, mean(AP));
myLogInfo('%s <= %g (upper bound)', APname, mean(APUB));
myLogInfo('%s >= %g (lower bound)', APname, mean(APLB));
toc(t0);
res = [mean(AP), mean(APUB), mean(APLB)];
end


function AP = get_AP(l)
cl = cumsum(l);
pl = cl ./ (1:length(l));
if sum(l) ~= 0
    rl = cl ./ sum(l);
else
    rl = zeros(1, length(cl));
end
drl = [0, diff(rl)];
AP = sum(drl .* pl);            
end
