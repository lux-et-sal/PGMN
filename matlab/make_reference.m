%% make_reference.m
% =====================================================================
% PGMN: write expected_outputs/metrics_reference.json.
%
% WHERE THE REFERENCE COMES FROM
%   Not from runExample_quick.m. If the package generated its own target it
%   would pass by construction and the check would prove nothing. The
%   reference is read from the production evaluation of the same patch:
%
%     accuracy     performance.*_per_grid inside the trained model file,
%                  written by the production evaluation code
%     uncertainty  recomputed from Patch_<NNNN>/predictions.mat of the
%                  shipped patch with uncertainty_metrics.m, which ports
%                  the definitions of the production metric code
%
%   The uncertainty half is recomputed rather than copied because the
%   published maps average over overlapping patches: 53 % of land cells
%   belong to two or more. Inside one patch's box those cells are blends
%   of that patch and its neighbors, which is right for a global map and
%   wrong as a single-patch target. Recomputing on the shipped patch alone
%   gives an unblended figure.
%
% CONVENTION
%   Every reported number is a median across scored cells, with the
%   interquartile range alongside. A cell is scored if it carries at least
%   ten paired observations in the evaluation window. See metrics.m for why
%   the alternative -- pooling all (cell, day) pairs -- is not used.
%
% TOLERANCES
%   Set from measured spread, not from taste. On the reference machine the
%   per-cell arrays agree with production to 7.6e-6 (GPU), and a CPU-only
%   run of the same package moves the reported medians by at most 1.9e-5
%   (per-cell worst case 9.7e-5). The tolerances below sit roughly fifty to
%   a hundred times above that: loose enough to survive a different BLAS or
%   an absent GPU, tight enough to fail on any substantive change.
%
% USAGE
%   >> make_reference           % preview
%   >> make_reference(true)     % write expected_outputs/metrics_reference.json
% =====================================================================

function make_reference(doWrite)

if nargin < 1 || isempty(doWrite), doWrite = false; end

%% USER PATHS ---------------------------------------------------------
BASE = 'C:/Users/Administrator/Desktop/WBM-LSTM_satellite_36km_EASE';
RDIR = fullfile(BASE, 'outputs', 'MDN-ConvLSTM', ...
                'outputs_individual_patches_QC_64x64_open_loop_TO_V1');

this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
addpath(this_dir);

DATA_FILE = fullfile(repo_root, 'data',            'sample_patch.mat');
OUT_JSON  = fullfile(repo_root, 'expected_outputs','metrics_reference.json');
MDN_FILE  = pick_one(fullfile(repo_root, 'pretrained', 'best_model_M*.mat'), ...
                     'Run trimBestModel(true) first.');

fprintf('========================================\n');
fprintf('PGMN: building expected_outputs/metrics_reference.json\n');
if ~doWrite
    fprintf('*** preview -- nothing is written. Run make_reference(true) ***\n');
end
fprintf('========================================\n\n');

assert(exist(DATA_FILE,'file') == 2, 'Run extract_sample_patch first.');

S = load(DATA_FILE, 'meta', 'split');
B = load(MDN_FILE,  'M', 'performance');
P = B.performance;

PDIR = fullfile(RDIR, sprintf('Patch_%04d', S.meta.patch_idx));
assert(exist(PDIR, 'dir') == 7, 'Production patch folder not found: %s', PDIR);

%% [1] Accuracy, straight from the production per-cell arrays ---------
fprintf('[1] Accuracy from performance.*_per_grid (%d cells)\n', numel(P.kge_mdn_per_grid));
acc = struct();
acc.wbm_baseline        = summarize4(P.kge_wbm_per_grid,  P.rmse_wbm_per_grid, ...
                                     P.r_wbm_per_grid,    P.bias_wbm_per_grid);
acc.pgmn_reconstruction = summarize4(P.kge_mdn_per_grid,  P.rmse_mdn_per_grid, ...
                                     P.r_mdn_per_grid,    P.bias_mdn_per_grid);
report('WBM baseline',        acc.wbm_baseline);
report('PGMN reconstruction', acc.pgmn_reconstruction);

%% [2] Uncertainty, recomputed on this patch alone --------------------
fprintf('\n[2] Uncertainty from Patch_%04d/predictions.mat\n', S.meta.patch_idx);
D = load(fullfile(PDIR, 'predictions.mat'), 'smap_true', 'smap_pred', 'residual_std');
% smap_true is already NaN wherever SMAP was not observed, so the
% observation mask is exactly its finite support.
umask = ~isnan(D.smap_true);
U = uncertainty_metrics(D.smap_true, D.smap_pred, D.residual_std, umask);

ufields = {'PICP95','q','MPIW','MAE','CRPS','MA','SB','CRPS_reduction_pct'};
unc = struct();
for i = 1:numel(ufields)
    k = ufields{i};
    unc.(k)       = round(U.(k), 6);
    unc.([k '_iqr']) = round(U.([k '_iqr']), 6);
    fprintf('    %-20s %+10.6f  [%+.6f, %+.6f]\n', k, U.(k), ...
            U.([k '_iqr'])(1), U.([k '_iqr'])(2));
end
fprintf('    cells scored         %d\n', U.n_grid);
assert(U.n_grid == numel(P.kge_mdn_per_grid), ...
    ['Accuracy and uncertainty disagree on the number of scored cells ' ...
     '(%d vs %d). They must describe the same cells.'], ...
    U.n_grid, numel(P.kge_mdn_per_grid));

%% [3] Assemble -------------------------------------------------------
sp = S.split;
ref = struct();
ref.x_comment = ['Reference values for PGMN patch ' num2str(S.meta.patch_idx) '. ' ...
    'Accuracy is copied from the production evaluation; uncertainty is ' ...
    'recomputed on this patch alone with the production definitions. Every entry is ' ...
    'a median across scored cells; *_iqr is [p25, p75].'];
ref.model      = 'PGMN';
ref.patch_idx  = S.meta.patch_idx;
ref.grid       = S.meta.grid;
ref.row_range  = S.meta.row_range;
ref.col_range  = S.meta.col_range;
ref.lat_range  = round(double(S.meta.lat_range), 4);
ref.lon_range  = round(double(S.meta.lon_range), 4);

ref.calibration_period = struct('steps', sp.T_calibration, ...
    'train_steps', sp.n_train, 'valid_steps', sp.n_valid, ...
    'note', 'train fits weights; valid drives early stopping only');
ref.evaluation_period = struct( ...
    'block_days',  sp.T_evaluation, ...
    'warmup_days', sp.seq_len - 1, ...
    'scored_days', sp.n_evaluation, ...
    'start', datestr(sp.scored_first_date, 'yyyy-mm-dd'), ...
    'end',   datestr(sp.scored_last_date,  'yyyy-mm-dd'), ...
    'note', 'the first seq_len-1 steps of the block are ConvLSTM warm-up and are not scored');

ref.convention = ['grid-wise median across scored cells; a cell is scored if it ' ...
    'has at least 10 paired observations in the evaluation window'];
ref.n_grid = U.n_grid;
ref.accuracy = acc;
ref.uncertainty = unc;

ref.model_selection = struct('M', double(B.M), ...
    'num_params_k', double(P.num_params), ...
    'AIC', round(double(P.AIC), 2), ...
    'NLL_calibration', round(double(P.NLL_cali), 6), 'n_calibration', double(P.n_cali), ...
    'NLL_valid',       round(double(P.NLL_valid), 6), 'n_valid',       double(P.n_valid), ...
    'NLL_evaluation',  round(double(P.NLL_test),  6), 'n_evaluation',  double(P.n_test), ...
    'note', 'AIC = 2k + 2*NLL_calibration*n_calibration; k counts Weights and Bias only');

ref.tolerance = struct('KGE', 1e-3, 'RMSE', 1e-4, 'R', 1e-3, 'Bias', 1e-4, ...
    'PICP95', 5e-3, 'q', 1e-3, 'MPIW', 1e-4, 'MAE', 1e-4, 'CRPS', 1e-4, ...
    'MA', 5e-3, 'SB', 5e-3, 'CRPS_reduction_pct', 1e-1, ...
    'x_comment', ['Absolute tolerance per metric. Measured spread: 7.6e-6 ' ...
                  'against production on GPU, at most 1.9e-5 between a GPU ' ...
                  'and a CPU run of this package. Most bounds are 50-100x ' ...
                  'that, to absorb a different BLAS or an absent GPU. ' ...
                  'PICP95, MA and SB get 5e-3 because they are counting ' ...
                  'statistics: one comparison e <= z*sigma flipping in a ' ...
                  'cell with 200 observations moves that cell by 1/200, so ' ...
                  'they are quantized in a way the continuous metrics are not.']);

%% [4] Write ----------------------------------------------------------
txt = jsonencode(ref, 'PrettyPrint', true);
txt = strrep(txt, '"x_comment"', '"_comment"');

if ~doWrite
    fprintf('\n[4] Preview of %s\n\n%s\n', OUT_JSON, txt);
    fprintf('Run make_reference(true) to write.\n');
    fprintf('========================================\n');
    return;
end

od = fileparts(OUT_JSON);
if ~exist(od, 'dir'); mkdir(od); end
if exist(OUT_JSON, 'file') == 2
    bak = [OUT_JSON '.bak_' datestr(now, 'yyyymmdd_HHMMSS')];
    movefile(OUT_JSON, bak);
    fprintf('\n  existing file preserved as %s\n', bak);
end
fid = fopen(OUT_JSON, 'w');  assert(fid > 0, 'Cannot write %s', OUT_JSON);
fwrite(fid, txt);  fclose(fid);
fprintf('  written %s (%d bytes)\n', OUT_JSON, numel(txt));
fprintf('========================================\n');

end

%% ---------------------------------------------------------------------
function f = pick_one(pattern, hint)
    d = dir(pattern);
    assert(~isempty(d), '%s not found. %s', pattern, hint);
    assert(numel(d) == 1, ...
        '%d files match %s -- exactly one model may be shipped.', numel(d), pattern);
    f = fullfile(d(1).folder, d(1).name);
end

function s = summarize4(kge, rmse, r, bias)
    s = struct();
    names = {'KGE','RMSE','R','Bias'};
    vals  = {kge,  rmse,  r,  bias};
    for i = 1:numel(names)
        v = double(vals{i}(:));  v = v(isfinite(v));
        s.(names{i}) = round(median(v), 6);
        s.([names{i} '_iqr']) = round([prctile(v,25), prctile(v,75)], 6);
    end
end

function report(label, s)
    fprintf('    --- %s ---\n', label);
    f = {'KGE','RMSE','R','Bias'};
    for i = 1:numel(f)
        q = s.([f{i} '_iqr']);
        fprintf('    %-5s %+10.6f  [%+.6f, %+.6f]\n', f{i}, s.(f{i}), q(1), q(2));
    end
end
