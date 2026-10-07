%% extract_wbm_params.m
% =====================================================================
% PGMN: cut the per-cell WBM parameters (alpha, Z, beta) of the sample
% patch from the global WBM result file (not part of this repository).
% A normal user does not need to run this. See SETUP_NOTES.md.
%
% Writes pretrained/wbm_params_patch<N>.mat. N and the row/column extent
% are read from data/sample_patch.mat, so run extract_sample_patch first.
%
% runExample_quick.m does not use this file. The WBM baseline is already
% in channel 2 of data/sample_patch.mat.
%
% SAVED FIELDS
%   alpha          slope of the loss function above theta_up      [1/day]
%   Z              effective depth (Delta Z)                      [mm]
%   beta           quantile of the dry-down limb regression       [0..1]
%   modeling_mask  cells where all three are finite
%   kge_wbm / rmse_wbm / r_wbm / bias_wbm
%                  WBM skill from the WBM run, for context only. These
%                  are not reproduction targets. Use expected_outputs/.
%
% The parameters are defined in the paper (Section 2.2.2).
%
% USER PATHS
%   Edit the block below to match your machine before running.
% =====================================================================

clear; clc;

%% USER PATHS ---------------------------------------------------------
BASE     = 'C:/Users/Administrator/Desktop/WBM-LSTM_satellite_36km_EASE';
WBM_FILE = fullfile(BASE, 'outputs', 'WBM', 'Open-Loop_TO', 'WBM_Global_Results_QC.mat');

this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
DATA_FILE = fullfile(repo_root, 'data', 'sample_patch.mat');

%% --------------------------------------------------------------------
fprintf('========================================\n');
fprintf('PGMN: extracting the sample patch WBM parameters\n');
fprintf('========================================\n\n');

assert(exist(WBM_FILE, 'file')==2, 'WBM result file not found: %s', WBM_FILE);
assert(exist(DATA_FILE,'file')==2, ...
    'Run extract_sample_patch first -- this script takes its extent from it.');

% Patch extent comes from the shipped data.
Sm = load(DATA_FILE, 'meta');
PATCH_IDX = Sm.meta.patch_idx;
ROW_RANGE = Sm.meta.row_range(1):Sm.meta.row_range(2);
COL_RANGE = Sm.meta.col_range(1):Sm.meta.col_range(2);
OUT_FILE  = fullfile(repo_root, 'pretrained', ...
                     sprintf('wbm_params_patch%d.mat', PATCH_IDX));

outdir = fileparts(OUT_FILE);
if ~exist(outdir, 'dir'); mkdir(outdir); end

m  = matfile(WBM_FILE);
sf = m.scale_factors;
fv = double(sf.fillValue);

a_int = double(m.Results_a_int16(ROW_RANGE, COL_RANGE));
Z_int = double(m.Results_Z_int16(ROW_RANGE, COL_RANGE));
b_int = double(m.Results_b_int16(ROW_RANGE, COL_RANGE));

alpha = single(a_int / double(sf.a_scaleFactor));   alpha(a_int==fv) = NaN;
Z     = single(Z_int / double(sf.Z_scaleFactor));   Z    (Z_int==fv) = NaN;
beta  = single(b_int / double(sf.b_scaleFactor));   beta (b_int==fv) = NaN;

modeling_mask = ~isnan(alpha) & ~isnan(Z) & ~isnan(beta);

%% WBM per-cell skill, for context only -------------------------------
kge_wbm  = deq(m.Results_KGE_int16 (ROW_RANGE, COL_RANGE), sf.KGE_scaleFactor,  fv);
rmse_wbm = deq(m.Results_RMSE_int16(ROW_RANGE, COL_RANGE), sf.RMSE_scaleFactor, fv);
r_wbm    = deq(m.Results_R_int16   (ROW_RANGE, COL_RANGE), sf.R_scaleFactor,    fv);
bias_wbm = deq(m.Results_Bias_int16(ROW_RANGE, COL_RANGE), sf.Bias_scaleFactor, fv);

meta = struct( ...
    'model',       'PGMN', ...
    'patch_idx',   PATCH_IDX, ...
    'row_range',   [ROW_RANGE(1), ROW_RANGE(end)], ...
    'col_range',   [COL_RANGE(1), COL_RANGE(end)], ...
    'grid',        'EASE-Grid 2.0 Global M36 (36 km equal-area)', ...
    'fitted_on',   'Calibration period only (2015-04-02 to 2022-12-31)', ...
    'fitted_by',   'production WBM calibration (main_wbm_open_loop.m, not shipped)', ...
    'on_run_path', false, ...
    'description', ['Per-cell WBM parameters. Documentation of the physics ' ...
                    'behind channel 2 of sample_patch.mat; not used by ' ...
                    'runExample_quick.m.'], ...
    'skill_caveat',['kge/rmse/r/bias_wbm are scored on the full 1004-day ' ...
                    'evaluation block, not the 995 days PGMN reports.'], ...
    'source_wbm',  basename(WBM_FILE), ...
    'created_at',  datestr(now, 'yyyy-mm-dd HH:MM:SS'));

save(OUT_FILE, 'alpha', 'Z', 'beta', 'modeling_mask', ...
     'kge_wbm', 'rmse_wbm', 'r_wbm', 'bias_wbm', 'meta', '-v7');

fprintf('Saved %s\n\n', OUT_FILE);
fprintf('  patch %d : rows %d:%d, cols %d:%d\n', PATCH_IDX, ...
    ROW_RANGE(1), ROW_RANGE(end), COL_RANGE(1), COL_RANGE(end));
fprintf('  modeled cells : %d / %d\n', sum(modeling_mask(:)), numel(modeling_mask));
fprintf('  alpha  [%.4f, %.4f]\n', min(alpha(modeling_mask)), max(alpha(modeling_mask)));
fprintf('  Z      [%.2f, %.2f] mm\n', min(Z(modeling_mask)),  max(Z(modeling_mask)));
fprintf('  beta   [%.4f, %.4f]\n', min(beta(modeling_mask)),  max(beta(modeling_mask)));
fprintf('  WBM KGE median (context only) : %.4f\n', median(kge_wbm(modeling_mask), 'omitnan'));
fprintf('========================================\n');

%% local ---------------------------------------------------------------
function v = deq(x_int, scale, fv)
    x_int = double(x_int);
    v = single(x_int / double(scale));
    v(x_int == fv) = NaN;
end

function s = basename(p)
    [~, n, e] = fileparts(p);
    s = [n, e];
end
