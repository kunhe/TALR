function [top] = ndcg_forward(layer, bot, top)
Y = squeeze(layer.class); % Nx1
X = squeeze(bot.x);  % 1x1xBxN -> BxN, raw scores (logits) for each bit
[nbits, N] = size(X);

opts  = layer.opts;
onGPU = numel(opts.gpus) > 0;

% get NxN affinity matrix
if ~opts.unsupervised, 
    assert(size(Y, 1) == N); 
    Aff = affinity_multlv(Y, Y, X, X, opts);
else
    assert(isvector(Y));
    assert(length(Y) == N);
    try
        Aff = opts.Aff(Y, Y) > 0;
    catch
        Aff = zeros(N, 'logical');
    end
end
V    = unique(Aff);
Naff = numel(V);
Gain = single(2.^Aff - 1);
Gns  = single(2.^V - 1);
vInd = cell(1, Naff);
for v = 1:Naff
    % find inds of this affinity value
    vInd{v} = (Aff == V(v));
    vInd{v}(logical(eye(N))) = false;
end
Discount = 1 ./ log2((1:N)' + 1);

% compute distances from relaxed hash codes
Phi  = 2 * sigmf(X, [opts.gamma_p 0]) - 1;
Dist = (nbits - phi' * phi) / 2;   

% histogram binning
histW = opts.nbits / opts.nbins;
histC = 0: histW: opts.nbits;
L     = length(histC); 
pulse = cell(1, L);
c_dv  = zeros(N, L, Naff);
if onGPU, 
    c_dv = gpuArray(c_dv); 
end
for l = 1:L
    pulse{l} = triPulse(hdist, Cntrs(l), Delta);  % NxN
    for v = 1:Naff
        c_dv(:, l, v) = sum(pulse{l} .* vInd{v}, 2);  % Nx1
    end
end
c_d = sum(c_dv, 3);     % NxL
C_d = cumsum(c_d, 2);   % NxL
C_1d = circshift(C_d, 1, 2);  % C_{d-1}, NxL
C_1d(:, 1) = 0;
Cbar = C_1d + (c_d+1)/2 + 1;

% NDCG
Ghat = 0;  % NxL
for v = 1:Naff
    Ghat = Ghat + Gns(v) * c_dv(:, :, v);
end
DCG  = sum(Ghat ./ log2(Cbar), 2);
DCGi = sort(Gain, 2, 'descend') * Discount;
NDCG = DCG ./ DCGi;
NDCG(DCGi == 0) = 1;

% loss
top.x = sum(NDCG);
top.aux = [];
top.aux.vInd  = vInd;
top.aux.Phi   = Phi;
top.aux.Dist  = Dist;
top.aux.histW = histW;
top.aux.histC = histC;
top.aux.Cbar  = Cbar;
top.aux.Gns   = Gns;
top.aux.Ghat  = Ghat;
top.aux.DCGi  = DCGi;
end
