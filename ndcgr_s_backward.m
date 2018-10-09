function [bot] = ndcglb_backward(layer, bot, top)
% Backprop for NDCGr, the continuous relaxation of tie-aware NDCG
% (simplified version)
%
X = squeeze(bot.x);  % nbitsxN
[nbits, N] = size(X);

% recover saved variables
opts  = layer.opts;
onGPU = numel(opts.gpus) > 0;
vInd  = top.aux.vInd;
Phi   = top.aux.Phi;
Dist  = top.aux.Dist;
histW = top.aux.histW;
histC = top.aux.histC;
Cbar  = top.aux.Cbar;
Gns   = top.aux.Gns;
Ghat  = top.aux.Ghat;
DCGi  = top.aux.DCGi;
L     = length(histC);
Naff  = length(vInd);

% 1. d(NDCGr_s)/d(c_d,v)
d_NDCG_c = cell(1, Naff);
for v = 1:Naff
    b = 1/log(2) * Ghat ./ log2(Cbar).^2 ./ Cbar;
    t = Gns(v) ./ log2(Cbar) - b/2 ...  % diagonal part
        - b * triu(ones(L), 1)';  % off-diagonal part
    % normalize by ideal DCG
    t = bsxfun(@rdivide, t, DCGi); 
    t(isnan(t)|isinf(t)) = 0;
    d_NDCG_c{v} = t;
end

% 3. d(NDCGr_s)/d(Phi)
d_NDCG_Phi = zeros(nbits, N);
if onGPU, d_NDCG_Phi = gpuArray(d_NDCG_Phi); end
for l = 1:L
    % NxN matrix of delta'(i, j, l) for fixed l
    dpulse = triPulseDeriv(Dist, histC(l), histW);  % NxN
    sumA = 0;
    for v = 1:Naff
        av = diag(d_NDCG_c{v}(:, l));
        bv = dpulse .* vInd{v};
        Av = av * bv + bv * av;
        sumA = sumA + Av;
    end
    d_NDCG_Phi = d_NDCG_Phi - 0.5 * Phi * sumA;
end

% 4. d(DCGLB)/d(x)
% completing the chain rule: tanh relaxation
% Note: tanh(x) = 2*sigmoid(2x)-1
sigm     = (Phi + 1) / 2;
d_Phi_x  = 2 .* sigm .* (1-sigm) * opts.gamma_p;  % nbitsxN
d_NDCG_x = -d_NDCG_Phi .* d_Phi_x;

% 5. final
bot.dzdx = zeros(size(bot.x), 'single');
if onGPU, bot.dzdx = gpuArray(bot.dzdx); end
bot.dzdx(1, 1, :, :) = single(d_NDCG_x);
end
