%% runExample_quick.m
% =====================================================================
% PGMN -- reproduce the reported evaluation-period numbers for the shipped
% sample patch in under a minute.
%
% WHAT IT DOES
%   1. loads data/sample_patch.mat (evaluation block, quantized)
%   2. rebuilds the trained network from pretrained/best_model_M*.mat
%   3. runs inference over the 995 scored days
%   4. scores accuracy grid-wise (median with IQR), for PGMN and for the
%      WBM baseline it corrects
%   5. scores the predictive distribution: coverage, sharpness, CRPS
%   6. writes outputs/ for verify_reproduction.m and plotReproduction.m
%
% WHAT IT DOES NOT DO
%   It does not run the water balance model. The WBM baseline was produced
%   once by the authors' production code (not part of this repository),
%   fitted on the calibration period, and travels inside channel 2 of the
%   sample patch. See extract_wbm_params.m for the physics parameters
%   behind that channel and wbm_simulate.m for the model itself.
%
%   It also does not train. Training the full 83-patch domain is out of
%   scope for this repository.
%
% Then:
%   >> cd ../expected_outputs
%   >> verify_reproduction
% =====================================================================

clear; clc;

%% Paths
this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
addpath(this_dir);                       % ConvLSTMLayer must be resolvable

DATA_FILE = fullfile(repo_root, 'data', 'sample_patch.mat');
OUT_DIR   = fullfile(repo_root, 'outputs');
if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

assert(exist(DATA_FILE,'file')==2, ...
    'data/sample_patch.mat missing -- see SETUP_NOTES.md');
md = dir(fullfile(repo_root, 'pretrained', 'best_model_M*.mat'));
assert(numel(md)==1, ...
    'Expected exactly one pretrained/best_model_M*.mat, found %d -- see SETUP_NOTES.md', ...
    numel(md));
MDN_FILE = fullfile(md(1).folder, md(1).name);

setSeed(42);

fprintf('========================================\n');
fprintf('PGMN runExample_quick\n');
fprintf('========================================\n\n');

%% [1/5] Load ---------------------------------------------------------
fprintf('[1/5] Loading\n');
S = load(DATA_FILE);
[net, M, normParams, model] = mdn_load_model(MDN_FILE);

sp = S.split;
fprintf('  patch %d       : rows %d-%d, cols %d-%d  (%.2f-%.2f N, %.2f-%.2f E)\n', ...
    S.meta.patch_idx, S.meta.row_range(1), S.meta.row_range(2), ...
    S.meta.col_range(1), S.meta.col_range(2), ...
    S.meta.lat_range(1), S.meta.lat_range(2), ...
    S.meta.lon_range(1), S.meta.lon_range(2));
fprintf('  grid          : %s\n', S.meta.grid);
fprintf('  mixture       : M = %d, k = %d weights counted for AIC\n', M, model.performance.num_params);
fprintf('  Calibration   : %d steps (train %d + valid %d)\n', ...
    sp.T_calibration, sp.n_train, sp.n_valid);
fprintf('  Evaluation    : %d steps scored, %s to %s\n\n', ...
    sp.n_evaluation, datestr(sp.scored_first_date), datestr(sp.scored_last_date));

%% [2/5] Inference ----------------------------------------------------
fprintf('[2/5] Inference\n');
t0 = tic;
R = mdn_reconstruct(net, M, normParams, S, sp.seq_len);   % GPU if available
fprintf('  wall time: %.1f sec\n\n', toc(t0));

%% [3/5] Accuracy, grid-wise -----------------------------------------
fprintf('[3/5] Accuracy (median across scored cells, IQR in brackets)\n');
metrics_mdn = metrics(R.smap_true, R.smap_pred, R.mask);
metrics_wbm = metrics(R.smap_true, R.wbm,       R.mask);

fprintf('  scored cells : %d\n', metrics_mdn.n_grid);
fprintf('\n  --- WBM baseline ---\n');       print_metric(metrics_wbm);
fprintf('\n  --- PGMN reconstruction ---\n'); print_metric(metrics_mdn);

% Per cell first, then the median -- the same order the tables use. Taking
% a ratio of two medians instead would not describe any actual cell.
fprintf('\n  --- Improvement (per cell, then median) ---\n');
dk = metrics_mdn.per_grid.KGE    - metrics_wbm.per_grid.KGE;
du = metrics_mdn.per_grid.ubRMSE - metrics_wbm.per_grid.ubRMSE;
pu = 100 * (1 - metrics_mdn.per_grid.ubRMSE ./ metrics_wbm.per_grid.ubRMSE);
fprintf('  delta KGE        = %+.4f\n', median(dk(isfinite(dk))));
fprintf('  delta ubRMSE     = %+.4f  (m^3 m^-3)\n', median(du(isfinite(du))));
fprintf('  ubRMSE reduction = %+.1f %%\n\n', median(pu(isfinite(pu))));

%% [4/5] Predictive distribution --------------------------------------
fprintf('[4/5] Predictive distribution\n');
unc = uncertainty_metrics(R.smap_true, R.smap_pred, R.residual_std, R.mask);
print_unc(unc);

%% [5/5] Save ---------------------------------------------------------
fprintf('\n[5/5] Saving\n');
save(fullfile(OUT_DIR, 'metrics_latest.mat'), ...
     'metrics_wbm', 'metrics_mdn', 'unc', '-v7.3');
save(fullfile(OUT_DIR, 'reconstruction.mat'), '-struct', 'R', '-v7.3');
fprintf('  outputs/metrics_latest.mat\n');
fprintf('  outputs/reconstruction.mat\n\n');

fprintf('========================================\n');
fprintf('Done. To check reproduction:\n');
fprintf('  >> cd ../expected_outputs\n');
fprintf('  >> verify_reproduction\n');
fprintf('========================================\n');

%% local --------------------------------------------------------------
function print_metric(m)
    f = {'Bias','ubRMSE','R','KGE'};
    for i = 1:numel(f)
        q = m.([f{i} '_iqr']);
        fprintf('  %-6s = %+9.6f   [%+.6f, %+.6f]\n', f{i}, m.(f{i}), q(1), q(2));
    end
end

function print_unc(u)
    fprintf('  scored cells : %d\n', u.n_grid);
    rows = {'PICP95', 'PICP_95    coverage of the nominal 95% interval'; ...
            'q',      'q          sqrt(E[(e/sigma)^2]), 1 = right-sized'; ...
            'MPIW',   'MPIW       interval width  (m^3 m^-3)'; ...
            'MAE',    'MAE        point-prediction error (m^3 m^-3)'; ...
            'CRPS',   'CRPS       distributional score (m^3 m^-3)'; ...
            'MA',     'MA         mean |coverage - nominal| over 0.025-0.975'; ...
            'SB',     'SB         signed deviation; - = too narrow'};
    for i = 1:size(rows,1)
        k = rows{i,1};  q = u.([k '_iqr']);
        fprintf('  %-52s %+9.6f  [%+.6f, %+.6f]\n', rows{i,2}, u.(k), q(1), q(2));
    end
    q = u.CRPS_reduction_pct_iqr;
    fprintf('  %-52s %+9.4f  [%+.4f, %+.4f]\n', ...
        'CRPS reduction vs MAE (%), per-cell then median', ...
        u.CRPS_reduction_pct, q(1), q(2));
end
