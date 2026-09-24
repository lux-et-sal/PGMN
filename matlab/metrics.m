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
%             (default 10, the rule used by the production evaluation)
%
% Output struct
%   Bias / ubRMSE / R / KGE      median across scored cells (the four
%                                metrics of Table 5 in the paper)
%   Bias_iqr / ubRMSE_iqr / ...  [p25 p75] across scored cells
%   per_grid                     struct of [n,1] per-cell vectors
%   n_grid                       number of scored cells
%   row, col                     [n,1] indices of those cells
%
% WHY GRID-WISE AND NOT POOLED
%   Pooling every (cell, day) pair into one population and computing a
%   single metric weights each cell by how often the satellite happened to
%   observe it. Revisit frequency is a function of orbit geometry, not of
%   scientific interest, and it varies by a factor of about twenty across
%   the domain. Adjacent cells are also not independent, so a pooled
%   sample size overstates the effective one by orders of magnitude.
%
%   Computing one metric per cell first, then taking the median across
%   cells, removes both problems. EASE-Grid 2.0 M36 is equal-area, so the
%   cell median needs no latitude weighting -- on a lat/lon grid it would.
%
%   The paper reports median with the interquartile range, so this function
%   returns both. The mean is deliberately not offered: the per-cell
%   distributions are skewed and a mean would be dragged by a handful of
%   badly conditioned cells.
%
% CELL SELECTION
%   A cell is scored if it has at least min_obs paired observations. This
%   is the same rule the evaluation code uses, so the numbers here line up
%   with performance.*_per_grid in the pretrained model file. ubRMSE is the
%   RMSE after removing the bias, sqrt(RMSE^2 - Bias^2), computed per cell.

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
% Kling-Gupta efficiency, Gupta et al. (2009).
% No guards, matching the production evaluation: a degenerate cell yields
% NaN and is dropped by the summary.
    r     = corr(o, s);
    alpha = std(s)  / std(o);
    beta  = mean(s) / mean(o);
    k = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2);
end
