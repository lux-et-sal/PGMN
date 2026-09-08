function [qreg, ts_filled, met_val] = wbm_forwardSim(params, ts, lb, ...
                                                      delta_t, p1, p2, phi, ...
                                                      idx_validsm) %#ok<INUSD>
% wbm_forwardSim  Run the WBM forward (and backward) soil moisture simulation
%                 with prescribed parameters and return performance metrics.
%
% NOT ON THE RUN PATH. Documentation of the physics behind channel 2 of
% data/sample_patch.mat; runExample_quick.m ships that baseline rather than
% recomputing it. Port of WBM/S4_smfilgaps_val_open.m. See wbm_simulate.m
% for how the production run differs.
%
%   [qreg, ts_filled, met_val] = wbm_forwardSim(params, ts, lb, delta_t, ...
%                                               p1, p2, phi, idx_validsm)
%
% This is the simulator used by both calibration and reconstruction. It
% propagates soil moisture forward from the first valid SMAP observation
% using a piecewise loss function fit to the dry-down limbs, and back-fills
% the timesteps before that first observation through fixed-point iteration
% on the same balance equation. No optimization is performed here.
%
% Inputs
%   params      [alpha, Z, beta], 1x3
%               alpha   slope of the loss function above p2 (1/day)
%               Z       active soil column thickness (mm)
%               beta    quantile used to fit the limb regression (0..1)
%   ts          [datenum, precip(mm/day), sm(m^3/m^3)], [n,3]
%   lb          dry-down limb table from wbm_drydown, columns [sm_initial,
%               loss_rate], [m,2]
%   delta_t     time step in days (use 1 for daily)
%   p1, p2      lower / upper soil moisture of the central linear regime
%   phi         soil porosity (m^3/m^3)
%   idx_validsm unused; kept so the call signature matches the calibration
%               code. Pass [] or anything.
%
% Outputs
%   qreg        [slope; intercept] of the quantile regression on dry-down
%               limbs, 2x1
%   ts_filled   [precip, sm_obs, loss, sm_sim, qf], [n,5]
%               sm_sim is the simulated soil moisture series (this is the
%               WBM baseline used by PGMN). qf is a per-step quality flag.
%   met_val     [bias, RMSE, ubRMSE, R, KGE], 1x5
%               metrics computed against sm_obs at observed timesteps
%               (NaN if fewer than 120 paired observations).
%
% Notes
%   - met_val is provided for diagnostics only; PGMN computes its own
%     evaluation-period metrics in runExample_quick.m.
%   - This function is deterministic given its inputs.

    alpha = params(1);
    Z     = params(2);
    beta  = params(3);

    pp = ts(:, 2);  pp(isnan(pp)) = 0;
    sm = ts(:, 3);

    %% --- quantile regression on dry-down limbs ---
    qreg = wbm_quantreg(lb(:, 1), lb(:, 2), beta, 1);

    if any(isnan(qreg)) || any(isinf(qreg))
        qreg      = [NaN; NaN];
        ts_filled = [pp, sm, nan(size(sm)), nan(size(sm)), zeros(size(sm))];
        met_val   = nan(1, 5);
        return;
    end

    %% --- preallocate ---
    n   = numel(sm);
    sms = nan(n, 1);
    los = nan(n, 1);
    qfs = zeros(n, 1);

    idx = find(~isnan(sm));
    if isempty(idx)
        ts_filled = [pp, sm, los, sms, qfs];
        met_val   = nan(1, 5);
        return;
    end

    fst_idx = idx(1);

    %% --- forward simulation from the first observation ---
    sms(fst_idx) = sm(fst_idx);
    qfs(fst_idx) = 4;

    for j = fst_idx + 1 : n
        [loss, qf]   = sub_loss(alpha, Z, sms(j-1), pp(j-1), p1, p2, ...
                                delta_t, phi, qreg);
        sms(j)       = sms(j-1) + (pp(j-1) / Z - loss) * delta_t;
        qfs(j)       = qf;

        if sms(j) > phi
            sms(j) = phi;
            qfs(j) = qf + 10;
        elseif sms(j) <= 0.02
            sms(j) = 0.02;
            qfs(j) = qf + 20;
        end
        los(j-1) = loss;
    end

    %% --- backward back-fill before the first observation ---
    if fst_idx > 1
        max_iter  = 20;
        tolerance = 1e-4;

        for j = fst_idx - 1 : -1 : 1
            sm_guess  = sms(j+1);
            converged = false;

            for it = 1:max_iter %#ok<NASGU>
                [loss, ~] = sub_loss(alpha, Z, sm_guess, pp(j), p1, p2, ...
                                     delta_t, phi, qreg);
                if isnan(loss)
                    sm_new = NaN;
                    break;
                end
                sm_new = sms(j+1) - (pp(j) / Z - loss) * delta_t;
                if abs(sm_new - sm_guess) < tolerance
                    converged = true;
                    break;
                end
                sm_guess = sm_new;
            end

            if converged
                sms(j) = sm_new;
                if isnan(sms(j))
                    qfs(j) = 59;
                elseif sms(j) > phi
                    sms(j) = phi;     qfs(j) = 52;
                elseif sms(j) <= 0.02
                    sms(j) = 0.02;    qfs(j) = 53;
                else
                    qfs(j) = 50;
                end
                if ~isnan(sms(j))
                    [los(j), ~] = sub_loss(alpha, Z, sms(j), pp(j), p1, p2, ...
                                            delta_t, phi, qreg);
                end
            else
                sms(j) = NaN;
                qfs(j) = 59;
                los(j) = NaN;
            end
        end
    end

    ts_filled = [pp, sm, los, sms, qfs];

    %% --- diagnostic metrics (paired observations) ---
    paired = [sm, sms];
    paired = paired(~any(isnan(paired), 2), :);
    if size(paired, 1) >= 120
        obs   = paired(:, 1);
        sim   = paired(:, 2);
        bias  = mean(sim - obs);
        rmse  = sqrt(mean((sim - obs).^2));
        ubrm  = sqrt(max(rmse^2 - bias^2, 0));
        r     = corr(obs, sim);
        b_kge = mean(sim) / mean(obs);
        a_kge = std (sim) / std (obs);
        if isnan(r) || ~isfinite(b_kge) || ~isfinite(a_kge)
            kge = NaN;
        else
            kge = 1 - sqrt((r - 1)^2 + (b_kge - 1)^2 + (a_kge - 1)^2);
        end
        met_val = [bias, rmse, ubrm, r, kge];
    else
        met_val = nan(1, 5);
    end
end


%% ---------------------------------------------------------------------
function [loss, qf] = sub_loss(alpha, Z, sm0, pp_t, p1, p2, delta_t, phi, qreg)
% Piecewise loss function used by the WBM (Akbar et al., 2018):
%   - linear ramp 0.02..p1
%   - regression-defined linear regime p1..p2
%   - linear extrapolation p2..phi with slope alpha
%   - saturated regime above phi (mass balance falls back to precip flux)

    if any(isnan([sm0, p1, p2, phi, alpha, Z]))
        loss = 0;  qf = 0;  return;
    end

    Lmin = p1 * qreg(1) + qreg(2);
    Lmax = p2 * qreg(1) + qreg(2);
    if isnan(Lmin) || isnan(Lmax)
        loss = 0;  qf = 0;  return;
    end

    sm0(sm0 < 0.02) = 0.02;

    if     sm0 >= 0.02 && sm0 <  p1
        loss = ((sm0 - 0.02) * Lmin) / (p1 - 0.02);   qf = 1;
    elseif sm0 >= p1   && sm0 <  p2
        loss = sm0 * qreg(1) + qreg(2);               qf = 2;
    elseif sm0 >= p2   && sm0 <= phi
        loss = Lmax + alpha * (sm0 - p2);             qf = 3;
    elseif sm0 >  phi
        loss = pp_t / Z - (phi - sm0) / delta_t;      qf = 4;
    else
        loss = NaN;                                   qf = 0;
    end
end