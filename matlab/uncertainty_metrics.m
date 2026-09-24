function out = uncertainty_metrics(obs, pred, sigma, mask, min_obs)
% uncertainty_metrics  Grid-wise calibration diagnostics for PGMN.
%
%   out = uncertainty_metrics(obs, pred, sigma, mask)
%   out = uncertainty_metrics(obs, pred, sigma, mask, min_obs)
%
% Inputs
%   obs, pred  [H,W,T]  observation and point prediction, m^3/m^3
%   sigma      [H,W,T]  predicted total standard deviation, m^3/m^3
%   mask       [H,W,T]  logical, true where an observation exists
%   min_obs    minimum paired observations per cell (default 10)
%
% Output struct -- medians across scored cells, with *_iqr companions
%   PICP95     coverage of the nominal 95 % interval
%   MPIW       mean prediction interval width, m^3/m^3
%   q          sqrt(E[(e/sigma)^2]); equals 1 when sigma is right-sized
%   MAE        mean absolute error of the point prediction, m^3/m^3
%   CRPS       continuous ranked probability score, m^3/m^3
%   CRPS_reduction_pct   median over cells of 100*(1 - CRPS/MAE)
%   MA         mean |coverage - nominal| over a sweep of confidence levels
%   SB         mean  (coverage - nominal); positive = over-covering
%   KS         max  |coverage - nominal| over the same sweep
%   per_grid   struct of per-cell vectors
%   n_grid     number of scored cells
%
% HOW TO READ THESE
%   PICP95 alone is weak evidence: it is one point on the calibration curve
%   and the model was fitted with a likelihood, so landing near 0.95 is
%   partly built in. MA and SB sweep the whole curve (0.025 to 0.975) and
%   ask whether coverage tracks the nominal level everywhere, which a
%   single-level check cannot see. SB carries the sign: negative means the
%   intervals are too narrow, positive means too wide.
%
%   q = 1 follows from the definition of a standard deviation and needs no
%   normality assumption. q < 1 means the spread is larger than the errors
%   warrant, q > 1 means overconfidence.
%
%   CRPS is compared against MAE because CRPS collapses to MAE when the
%   forecast is a point mass. The reduction is therefore the value added by
%   issuing a distribution rather than a number. It is a median of per-cell
%   ratios, not a ratio of medians -- the two differ and only the former is
%   a statement about a typical cell.
%
% DEFINITIONS
%   Port of the definitions used by the authors' production metric code.
%   With e = |pred - obs| and w = e/sigma:
%       PICP95 = mean( e <= z95*sigma ),      z95 = 1.959963985
%       MPIW   = mean( 2*z95*sigma )
%       q      = sqrt( mean(w^2) )
%       CRPS   = mean( sigma * ( w(2*Phi(w)-1) + 2*phi(w) - 1/sqrt(pi) ) )
%   The CRPS expression is the closed form for a Gaussian predictive
%   distribution, evaluated on the total-variance summary of the mixture.
%
%   One deliberate difference from the production code: cells are required
%   to have at least min_obs observations, matching metrics.m, so the
%   accuracy and the calibration tables describe the same set of cells. The
%   production code keeps every cell with one observation because it
%   aggregates across all 83 patches.

    if nargin < 5 || isempty(min_obs), min_obs = 10; end

    z95 = 1.959963985;
    cg  = (0.025:0.05:0.975)';          % confidence sweep, K = 20
    zg  = sqrt(2) * erfinv(cg);         % z_c = norminv((1+c)/2)
    Kc  = numel(cg);

    obs   = double(obs);
    pred  = double(pred);
    sigma = double(sigma);
    mask  = logical(mask);

    [H, W, ~] = size(obs);

    picp = []; mpiw = []; qq = []; mae = []; crps = [];
    MA = []; SB = []; KS = []; nobs = []; rows = []; cols = [];
    PC_all = [];

    for i = 1:H
        for j = 1:W
            m = squeeze(mask(i, j, :));
            if ~any(m); continue; end

            o = squeeze(obs  (i, j, :));
            p = squeeze(pred (i, j, :));
            g = squeeze(sigma(i, j, :));

            keep = m & ~isnan(o) & ~isnan(p) & ~isnan(g) & g > 0;
            n = sum(keep);
            if n < min_obs; continue; end

            e = abs(p(keep) - o(keep));
            s = g(keep);
            w = e ./ s;

            Phi = 0.5 * (1 + erf(w / sqrt(2)));
            phi = exp(-0.5 * w.^2) / sqrt(2*pi);
            cr  = s .* ( w .* (2*Phi - 1) + 2*phi - 1/sqrt(pi) );

            picp(end+1,1) = mean(e <= z95 * s);                 %#ok<AGROW>
            mpiw(end+1,1) = mean(2 * z95 * s);                  %#ok<AGROW>
            mae (end+1,1) = mean(e);                            %#ok<AGROW>
            crps(end+1,1) = mean(cr);                           %#ok<AGROW>
            qq  (end+1,1) = sqrt(mean(w.^2));                   %#ok<AGROW>

            PC = zeros(1, Kc);
            for k = 1:Kc, PC(k) = mean(w <= zg(k)); end
            dev = PC - cg';
            MA(end+1,1) = mean(abs(dev));                       %#ok<AGROW>
            SB(end+1,1) = mean(dev);                            %#ok<AGROW>
            KS(end+1,1) = max(abs(dev));                        %#ok<AGROW>
            PC_all(end+1,:) = PC;                               %#ok<AGROW>

            nobs(end+1,1) = n;                                  %#ok<AGROW>
            rows(end+1,1) = i;                                  %#ok<AGROW>
            cols(end+1,1) = j;                                  %#ok<AGROW>
        end
    end

    % Value of issuing a distribution: per cell first, then the median.
    ok  = isfinite(crps) & isfinite(mae) & mae > 0;
    red = 100 * (1 - crps(ok) ./ mae(ok));

    out = struct();
    out.n_grid = numel(qq);
    out.row = rows;  out.col = cols;
    out.confidence_levels = cg';
    out.coverage_curve = PC_all;
    out.per_grid = struct('PICP95', picp, 'MPIW', mpiw, 'q', qq, ...
                          'MAE', mae, 'CRPS', crps, 'MA', MA, 'SB', SB, ...
                          'KS', KS, 'CRPS_reduction_pct', 100*(1 - crps./mae), ...
                          'n_obs', nobs);

    names = {'PICP95','MPIW','q','MAE','CRPS','MA','SB','KS'};
    vals  = {picp,    mpiw,  qq, mae,  crps,  MA,  SB,  KS};
    for k = 1:numel(names)
        v = vals{k};  v = v(isfinite(v));
        if isempty(v)
            out.(names{k}) = NaN;  out.([names{k} '_iqr']) = [NaN NaN];
        else
            out.(names{k}) = median(v);
            out.([names{k} '_iqr']) = [prctile(v,25), prctile(v,75)];
        end
    end

    if isempty(red)
        out.CRPS_reduction_pct = NaN;
        out.CRPS_reduction_pct_iqr = [NaN NaN];
    else
        out.CRPS_reduction_pct = median(red);
        out.CRPS_reduction_pct_iqr = [prctile(red,25), prctile(red,75)];
    end
end
