function out = metrics(obs, sim, mask, min_obs)
% metrics  Grid-wise accuracy metrics, summarized by median and IQR.
%
%   out = metrics(obs, sim, mask)
%   out = metrics(obs, sim, mask, min_obs)
%
% Inputs
%   obs, sim  [H,W,T]  observations and simulations, m^3/m^3
%   mask      [H,W,T]  logical, true where an observation exists
%   min_obs   minimum paired observations a cell needs to be scored
%             (default 10)
%
% Output struct
%   Bias / ubRMSE / R / KGE      median across scored cells
%   Bias_iqr / ubRMSE_iqr / ...  [p25 p75] across scored cells
%   per_grid                     struct of [n,1] per-cell vectors
%   n_grid                       number of scored cells
%   row, col                     [n,1] indices of those cells
%
% A cell is scored if it has at least min_obs paired observations. ubRMSE
% is computed per cell as sqrt(RMSE^2 - Bias^2). The metrics are defined
% in the paper (Section 2.4).

    if nargin < 4 || isempty(min_obs), min_obs = 10; end

    obs  = double(obs);
    sim  = double(sim);
    mask = logical(mask);

    [H, W, ~] = size(obs);

    ubrmse = []; r = []; kge = []; bias = []; nobs = [];
    rows = []; cols = [];

    for i = 1:H
        for j = 1:W
            m = squeeze(mask(i, j, :));
            if ~any(m); continue; end

            o = squeeze(obs(i, j, :));
            s = squeeze(sim(i, j, :));
            keep = m & ~isnan(o) & ~isnan(s);
            n = sum(keep);
            if n < min_obs; continue; end

            ok = o(keep);  sk = s(keep);

            b    = mean(sk - ok);
            rm   = sqrt(mean((ok - sk).^2));
            bias  (end+1,1) = b;                                %#ok<AGROW>
            ubrmse(end+1,1) = sqrt(max(rm^2 - b^2, 0));         %#ok<AGROW>
            r     (end+1,1) = corr(ok, sk);                     %#ok<AGROW>
            kge   (end+1,1) = local_kge(ok, sk);                %#ok<AGROW>
            nobs(end+1,1) = n;                                  %#ok<AGROW>
            rows(end+1,1) = i;                                  %#ok<AGROW>
            cols(end+1,1) = j;                                  %#ok<AGROW>
        end
    end

    out = struct();
    out.n_grid = numel(kge);
    out.row = rows;  out.col = cols;
    out.per_grid = struct('Bias', bias, 'ubRMSE', ubrmse, 'R', r, ...
                          'KGE', kge, 'n_obs', nobs);

    names = {'Bias', 'ubRMSE', 'R', 'KGE'};
    vals  = {bias,   ubrmse,   r,   kge};
    for k = 1:numel(names)
        v = vals{k};
        v = v(isfinite(v));
        if isempty(v)
            out.(names{k}) = NaN;
            out.([names{k} '_iqr']) = [NaN NaN];
        else
            out.(names{k}) = median(v);
            out.([names{k} '_iqr']) = [prctile(v, 25), prctile(v, 75)];
        end
    end
end

%% ---------------------------------------------------------------------
function k = local_kge(o, s)
% Kling-Gupta efficiency. A degenerate cell gives NaN and is dropped by
% the summary.
    r     = corr(o, s);
    alpha = std(s)  / std(o);
    beta  = mean(s) / mean(o);
    k = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2);
end
