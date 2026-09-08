function theta_WBM = wbm_simulate(smap, precip, porosity, time_datenum, ...
                                   valid_mask, alpha, Z, beta, modeling_mask)
% wbm_simulate  Run the WBM forward+backward simulation on every cell of a
%               patch using pre-fitted parameters. No optimization.
%
% ---------------------------------------------------------------------
% NOT ON THE RUN PATH. PGMN ships the WBM baseline already simulated, in
% channel 2 of data/sample_patch.mat, so runExample_quick.m never calls
% this function. It is kept as readable documentation of the physics that
% produced that channel.
%
% ⚠ IT IS NOT A DROP-IN REPLICA OF THE PRODUCTION RUN. Two differences:
%
%   (1) Dry-down segments. This function extracts them from the FULL
%       record. WBM/main_wbm_open_loop.m extracts them from the
%       CALIBRATION period only and then simulates forward over the full
%       record with those calibration-derived limbs. Fitting the loss
%       function on data that includes the evaluation window would leak.
%
%   (2) Admission thresholds. Production skips a cell unless it yields at
%       least 10 dry-down points and 15 limbs; this function only requires
%       the segment list to be non-empty, so it will attempt cells that
%       production declines.
%
% Running it will therefore give a series close to, but not identical
% with, the shipped baseline. If you need the production behavior, use
% WBM/main_wbm_open_loop.m with S2_smdd and S4_smfilgaps_val_open.
% ---------------------------------------------------------------------
%
%   theta_WBM = wbm_simulate(smap, precip, porosity, time_datenum, ...
%                            valid_mask, alpha, Z, beta, modeling_mask)
%
% This is a thin patch-level wrapper around wbm_drydown and wbm_forwardSim.
% For each cell that is flagged as runnable it (1) extracts dry-down segments
% from the soil moisture series, (2) propagates the WBM water balance forward
% from the first valid SMAP observation and back-fills the earlier timesteps,
% and (3) stores the simulated soil moisture series in the output array.
%
% The per-cell parameters (alpha, Z, beta) are taken from a previous global
% calibration of the WBM (see extract_wbm_params.m); this function never
% calls the optimizer. As a consequence the simulation is fully
% deterministic given its inputs.
%
% Inputs
%   smap          [H,W,T] single   SMAP soil moisture (m^3/m^3), NaN allowed
%   precip        [H,W,T] single   MSWEP precipitation (mm/day), NaN allowed
%   porosity      [H,W]   single   SMAP L4 porosity (m^3/m^3)
%   time_datenum  [T,1]   double   daily timestamps as MATLAB datenum
%   valid_mask    [H,W]   logical  cells that passed SMAP / WBM grid QC
%   alpha,Z,beta  [H,W]   single   pre-fitted per-cell WBM parameters
%   modeling_mask [H,W]   logical  cells with finite (alpha, Z, beta)
%
% Output
%   theta_WBM     [H,W,T] single   simulated soil moisture (m^3/m^3),
%                                  NaN at every cell where simulation was
%                                  not attempted or did not produce output
%
% Constants
%   The four constants below are fixed to the values used by the global
%   calibration (main_wbm_open_loop.m):
%     consec   = 3      minimum observations per dry-down segment
%     pp_min   = 0.5    precipitation threshold (mm) below which inter-
%                       observation rainfall is treated as zero
%     hour_dif = 96     maximum gap (hours) between two successive SMAP
%                       observations within one dry-down segment
%     delta_t  = 1      simulation time step in days
%
% Notes
%   - Cells with valid_mask=false, modeling_mask=false, or NaN porosity are
%     skipped silently and remain NaN in theta_WBM.
%   - A parfor loop is used; if no parallel pool is open MATLAB falls back
%     to serial execution automatically. With a small pool a 64x64 patch
%     over the full record runs in a few minutes; serial execution takes
%     several times longer.
%   - This function depends on wbm_drydown.m and wbm_forwardSim.m being on
%     the MATLAB path. wbm_forwardSim.m in turn depends on wbm_quantreg.m.

    %% --- constants (must match main_wbm_open_loop.m) -----------------
    consec   = 3;
    pp_min   = 0.5;
    hour_dif = 96;
    delta_t  = 1;

    %% --- preallocate output -----------------------------------------
    [H, W, T] = size(smap);
    theta_WBM = nan(H, W, T, 'single');

    %% --- decide which cells to run ----------------------------------
    Time_All = datetime(time_datenum, 'ConvertFrom', 'datenum');
    tt       = time_datenum(:);

    runnable      = valid_mask & modeling_mask & ~isnan(porosity);
    [rows, cols]  = find(runnable);
    n_cells       = numel(rows);

    fprintf('  wbm_simulate: %d cells to run (of %d in patch)\n', ...
            n_cells, H * W);

    if n_cells == 0
        fprintf('  wbm_simulate: nothing to do.\n');
        return;
    end

    % Slice arrays once for parfor efficiency: per-cell column vectors.
    sm_cells = zeros(T, n_cells, 'single');
    pp_cells = zeros(T, n_cells, 'single');
    for k = 1:n_cells
        sm_cells(:, k) = squeeze(smap  (rows(k), cols(k), :));
        pp_cells(:, k) = squeeze(precip(rows(k), cols(k), :));
    end
    phi_cells   = porosity(sub2ind([H, W], rows, cols));
    alpha_cells = alpha   (sub2ind([H, W], rows, cols));
    Z_cells     = Z       (sub2ind([H, W], rows, cols));
    beta_cells  = beta    (sub2ind([H, W], rows, cols));

    sim_cells = nan(T, n_cells, 'single');

    %% --- per-cell simulation ----------------------------------------
    parfor k = 1:n_cells
        vsm_p = sm_cells(:, k);
        gpm_p = pp_cells(:, k);
        gpm_p(isnan(gpm_p)) = 0;            % match wbm_forwardSim convention
        phi_p = phi_cells(k);

        try
            % (1) extract dry-down segments
            [paras_dd, lb, sm_outputs] = wbm_drydown(Time_All, gpm_p, vsm_p, ...
                                                      consec, pp_min, phi_p, hour_dif);

            % cells that produced no usable segments stay NaN
            if isempty(lb), continue; end
            idx_drydn = find(sm_outputs(:, 2) == 4);
            if isempty(idx_drydn), continue; end

            p1 = paras_dd(2);
            p2 = paras_dd(3);

            % (2) forward + backward simulation with the known parameters
            params = [alpha_cells(k), Z_cells(k), beta_cells(k)];
            LF     = [lb(:, 2), lb(:, 5)];     % [sm_initial, loss_rate]

            [~, ts_filled, ~] = wbm_forwardSim(params, ...
                                                [tt, gpm_p, vsm_p], ...
                                                LF, delta_t, p1, p2, ...
                                                phi_p, idx_drydn);

            % (3) store the simulated series (column 4 of ts_filled)
            sim_cells(:, k) = single(ts_filled(:, 4));
        catch
            % leave column as NaN; failure of one cell must not abort the patch
        end
    end

    %% --- scatter results back into the [H,W,T] grid ----------------
    for k = 1:n_cells
        theta_WBM(rows(k), cols(k), :) = sim_cells(:, k);
    end

    n_done = sum(any(~isnan(theta_WBM), 3) & runnable, 'all');
    fprintf('  wbm_simulate: %d / %d cells produced output\n', n_done, n_cells);
end