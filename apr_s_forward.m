function [top] = apr_s_forward(layer, bot, top)
% Forward computation for APr, the continuous relaxation of tie-aware AP 
% (simplified version)
%
Y = squeeze(layer.class); % Nx1
X = squeeze(bot.x);  % 1x1xBxN -> BxN, raw scores (logits) for each bit
[nbits, N] = size(X);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;

% get NxN affinity matrix
if opts.unsupervised
    assert(isvector(Y));
    assert(length(Y) == N);
    Aff = opts.Aff(Y, Y);
else
    assert(size(Y, 1) == N); 
    Aff = affinity_binary(Y, Y, X, X, opts);
end
% set diagonal to 0: x is not itself's neighbor
Xp = logical(Aff - diag(diag(Aff)));
Xn = ~Aff;
if onGPU
    Xp = gpuArray(Xp);
    Xn = gpuArray(Xn);
end

% compute distances from hash codes, with tanh relaxation
% Note: tanh(x) = 2 * sigmoid(2x) - 1
Phi  = 2 * sigmf(X, [opts.gamma_p 0]) - 1;
Dist = (nbits - Phi' * Phi) / 2;   

% build discrete distributions with differentiable histogram binning
histW = opts.nbits / opts.nbins;
histC = 0 : histW : opts.nbits;
histW = histW * opts.delta;
L  = length(histC); 
c  = zeros(N, L);   % distance histogram (continuous relaxation)
cp = zeros(N, L);   % positive histogram (continuous relaxation)
cn = zeros(N, L);   % negative histogram (continuous relaxation)
if onGPU
    c  = gpuArray(c);
    cp = gpuArray(cp);
    cn = gpuArray(cn);
end
% triangular pulse / linear interpolation
for l = 1:L
    pulse = triPulse(Dist, histC(l), histW);  % NxN
    cp(:, l) = sum(pulse .* Xp, 2);
    cn(:, l) = sum(pulse .* Xn, 2);
end
c  = cp + cn;
C  = cumsum(c, 2);  % cumulative histogram (continuous relaxation)
Cp = cumsum(cp, 2);
Cn = cumsum(cn, 2);

% common variables to reuse
C_1d  = circshift(C , 1, 2);  C_1d (:, 1) = 0;   % C_{d-1}
Cp_1d = circshift(Cp, 1, 2);  Cp_1d(:, 1) = 0;   % C_{d-1}^+
Cn_1d = circshift(Cn, 1, 2);  Cn_1d(:, 1) = 0;   % C_{d-1}^-
numer = Cp_1d + Cp + 1;
denom = C_1d  + C  + 1;

% compute simplified APr
APr_s = cp .* numer ./ denom;
APr_s = sum(APr_s, 2) ./ sum(Xp, 2);
APr_s(isnan(APr_s)) = 0;

% adjust for Delta scaling
APr_s = APr_s / opts.delta;

% loss
top.x = sum(APr_s);

% variables to be reused in backward
top.aux = [];
top.aux.Xp    = Xp;
top.aux.Xn    = Xn;
top.aux.Phi   = Phi;
top.aux.Dist  = Dist;
top.aux.histC = histC;
top.aux.histW = histW;
top.aux.c     = c;
top.aux.cp    = cp;
top.aux.cn    = cn;
top.aux.C     = C;
top.aux.Cp    = Cp;
top.aux.Cn    = Cn;
top.aux.numer = numer;
top.aux.denom = denom;

end
