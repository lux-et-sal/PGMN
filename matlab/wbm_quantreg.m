function p = wbm_quantreg(x, y, tau, order)
% wbm_quantreg  Polynomial quantile regression by minimizing the asymmetric
%               absolute-residual loss.
%
% NOT ON THE RUN PATH. Called only by wbm_forwardSim, which is itself kept
% as documentation -- see that file. Port of WBM/S5_quantreg.m.
%
%   p = wbm_quantreg(x, y, tau, order)
%
% Fits y = polyval(p, x) at the tau-th quantile (0 < tau < 1) using
% fminsearch. Used by wbm_forwardSim to derive the central linear regime
% of the WBM loss function from the dry-down limb cloud.
%
% Inputs
%   x      column vector of regressors (the limb initial soil moisture)
%   y      column vector of responses  (the limb loss rates)
%   tau    target quantile, 0 < tau < 1
%   order  polynomial order; defaults to 1 (linear) and is the only order
%          used in PGMN
%
% Output
%   p      polynomial coefficients, [order+1, 1]
%
% Origin
%   Adapted from QUANTREG by Aslak Grinsted (2008), MathWorks File Exchange.
%   Bootstrap confidence-interval branch and zero-intercept branch removed
%   for clarity; the deterministic fminsearch fit is unchanged.

    if nargin < 4 || isempty(order), order = 1; end

    if tau <= 0 || tau >= 1
        error('wbm_quantreg:tau', 'tau must be in (0, 1).');
    end
    if numel(y) ~= size(y, 1)
        error('wbm_quantreg:y', 'y must be a column vector.');
    end
    if size(x, 1) ~= size(y, 1)
        error('wbm_quantreg:size', 'x and y must have the same number of rows.');
    end

    x = double(x);
    y = double(y);

    %% --- design matrix (Vandermonde for univariate x) ---
    if size(x, 2) == 1
        X = ones(numel(x), order + 1);
        for k = 1:order
            X(:, order - k + 1) = x .^ k;
        end
    else
        X = x;       % multi-column x is treated as the design directly
    end

    %% --- fit ---
    rho   = @(r) sum(abs(r .* (tau - (r < 0))));
    p_ols = X \ y;                              % OLS warm start
    p     = fminsearch(@(p) rho(y - X * p), p_ols);
end