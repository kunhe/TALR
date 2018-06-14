function [bot] = apr_s_backward(layer, bot, top)
% Backprop for APr, the continuous relaxation of tie-aware AP 
% (simplified version)
%
X = squeeze(bot.x);  % nbitsxN
[nbits, N] = size(X);

% recover saved variables
opts  = layer.opts;
onGPU = numel(opts.gpus) > 0;
Xp    = top.aux.Xp;
Xn    = top.aux.Xn;
Phi   = top.aux.Phi;
Dist  = top.aux.Dist;
histW = top.aux.histW;
histC = top.aux.histC;
L     = length(histC);
c     = top.aux.c;
cp    = top.aux.cp;
cn    = top.aux.cn;
C     = top.aux.C;
Cp    = top.aux.Cp;
Cn    = top.aux.Cn;
numer = top.aux.numer;
denom = top.aux.denom;
denom2 = denom .* denom;

% 1. d(APr_s)/d(c+) in matrix form
% diagonal terms
d_AP_cp_dia = (2*Cp+1)./denom - cp.*numer./denom2;
% off-diagonal terms
d_AP_cp_off = cp .* (denom - numer) ./ denom2;
% combine
d_AP_cp = d_AP_cp_dia + d_AP_cp_off * triu(ones(L), 1)'; 
% normalize
d_AP_cp = bsxfun(@rdivide, d_AP_cp, sum(Xp, 2));
d_AP_cp(isnan(d_AP_cp)|isinf(d_AP_cp)) = 0;

% 2. d(APr_s)/d(c-)
% diagonal & off-diagonal terms (are the same)
d_AP_cn = -cp .* numer ./ denom2;
% combine
d_AP_cn = d_AP_cn * triu(ones(L))';
% normalize
d_AP_cn = bsxfun(@rdivide, d_AP_cn, sum(Xp, 2));
d_AP_cn(isnan(d_AP_cn)|isinf(d_AP_cn)) = 0;

% 3. d(APr_s)/d(Phi)
% advancing the chain rule with differentiable histogram binning
d_AP_Phi = zeros(nbits, N);
if onGPU
    d_AP_Phi = gpuArray(d_AP_Phi); 
end
for l = 1:L
    % NxN matrix of delta'(i, j, l) for fixed l
    dpulse = triPulseDeriv(Dist, histC(l), histW);  % NxN
    ddp = dpulse .* Xp;
    ddn = dpulse .* Xn;

    % A*B + B*A
    alpha_p = diag(d_AP_cp(:, l));  
    alpha_n = diag(d_AP_cn(:, l));  
    Ap = ddp * alpha_p + alpha_p * ddp;
    An = ddn * alpha_n + alpha_n * ddn;

    % accumulate gradient
    d_AP_Phi = d_AP_Phi - 0.5 * Phi * (Ap + An);
end

% 4. d(APr_s)/d(x)
% completing the chain rule: tanh relaxation
% Note: tanh(x) = 2*sigmoid(2x)-1
sigm    = (Phi + 1) / 2;
d_Phi_x = 2 .* sigm .* (1-sigm) * opts.gamma_p;  % nbitsxN
d_AP_x  = -d_AP_Phi .* d_Phi_x;  % MatConvNet does minimization

% 5. finalize
bot.dzdx = zeros(size(bot.x), 'single');
if onGPU
    bot.dzdx = gpuArray(bot.dzdx); 
end
bot.dzdx(1, 1, :, :) = single(d_AP_x);

end
