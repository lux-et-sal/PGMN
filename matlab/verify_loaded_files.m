%% verify_loaded_files.m
% =====================================================================
% PGMN: check that the three shipped files describe the same patch, the
% same period and the same model, before anything is run.
%
% This is a cross-consistency check, not a reproduction check. It catches
% the failure mode that matters when files are regenerated one at a time:
% a sample patch from one region paired with a model trained on another.
% Every number would still look plausible and every metric would be wrong.
%
% For the reproduction check itself, run runExample_quick.m and then
% expected_outputs/verify_reproduction.m.
%
% Run from matlab/ :
%   >> verify_loaded_files
% =====================================================================

clear; clc;

this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
addpath(this_dir);              % ConvLSTMLayer must be resolvable

DATA_FILE = fullfile(repo_root, 'data', 'sample_patch.mat');
assert(exist(DATA_FILE,'file')==2, 'data/sample_patch.mat missing.');

md = dir(fullfile(repo_root, 'pretrained', 'best_model_M*.mat'));
assert(numel(md)==1, 'Expected exactly one pretrained/best_model_M*.mat, found %d.', numel(md));
MDN_FILE = fullfile(md(1).folder, md(1).name);

wd = dir(fullfile(repo_root, 'pretrained', 'wbm_params_patch*.mat'));
assert(numel(wd)==1, 'Expected exactly one pretrained/wbm_params_patch*.mat, found %d.', numel(wd));
WBM_FILE = fullfile(wd(1).folder, wd(1).name);

S = load(DATA_FILE);
B = load(MDN_FILE);
P = load(WBM_FILE);

fprintf('========================================\n');
fprintf('PGMN file consistency check\n');
fprintf('========================================\n\n');
fprintf('  data/%s\n', 'sample_patch.mat');
fprintf('  pretrained/%s\n', md(1).name);
fprintf('  pretrained/%s\n\n', wd(1).name);

ok = true;

%% [1] Same patch -----------------------------------------------------
fprintf('[1] Same patch\n');
ok = eq_check('patch index (data vs WBM params)', ...
              S.meta.patch_idx, P.meta.patch_idx) && ok;
ok = eq_check('row range  (data vs model)', ...
              S.meta.row_range(:)', [double(B.patch_info.row_start), double(B.patch_info.row_end)]) && ok;
ok = eq_check('col range  (data vs model)', ...
              S.meta.col_range(:)', [double(B.patch_info.col_start), double(B.patch_info.col_end)]) && ok;
ok = eq_check('row range  (data vs WBM params)', ...
              S.meta.row_range(:)', double(P.meta.row_range(:)')) && ok;
ok = eq_check('col range  (data vs WBM params)', ...
              S.meta.col_range(:)', double(P.meta.col_range(:)')) && ok;
fprintf('    extent %d x %d, %.3f-%.3f N, %.3f-%.3f E\n\n', ...
    size(S.LAT,1), size(S.LAT,2), S.meta.lat_range(1), S.meta.lat_range(2), ...
    S.meta.lon_range(1), S.meta.lon_range(2));

%% [2] Same shapes ----------------------------------------------------
fprintf('[2] Shapes\n');
[Hs, Ws, Cx, Ts] = size(S.XVal);
ok = eq_check('input height', Hs, double(B.input_size(1))) && ok;
ok = eq_check('input width',  Ws, double(B.input_size(2))) && ok;
% The model takes four channels; three are stored, ResAvail is the fourth.
ok = eq_check('stored + ResAvail channels', Cx + 1, double(B.input_size(3))) && ok;
ok = eq_check('WBM parameter grid', [Hs Ws], size(P.alpha)) && ok;
ok = eq_check('target length', Ts, size(S.YVal_Direct, 3)) && ok;
ok = eq_check('mask length',   Ts, size(S.MaskVal, 3)) && ok;
fprintf('\n');

%% [3] Split ----------------------------------------------------------
fprintf('[3] Split\n');
sp = S.split;
ok = eq_check('evaluation block length', sp.T_evaluation, Ts) && ok;
ok = eq_check('scored + warm-up = block', sp.n_evaluation + sp.seq_len - 1, Ts) && ok;
ok = eq_check('train + valid consistency', ...
              sp.n_train, numel(sp.seq_len : floor(sp.T_calibration*sp.train_frac))) && ok;
fprintf('    calibration %d steps (train %d + valid %d)\n', ...
    sp.T_calibration, sp.n_train, sp.n_valid);
fprintf('    evaluation  %d of %d block days, %s to %s\n\n', ...
    sp.n_evaluation, sp.T_evaluation, ...
    datestr(sp.scored_first_date), datestr(sp.scored_last_date));

%% [4] Model ----------------------------------------------------------
fprintf('[4] Model\n');
nlearn = 0;
for i = 1:numel(B.learnables_vals), nlearn = nlearn + numel(B.learnables_vals{i}); end
ok = str_check('save method', B.save_method, 'learnables_direct') && ok;
fprintf('    M = %d, %d stored values, k = %d counted for AIC\n', ...
    B.M, nlearn, B.performance.num_params);

% The real test: can the network actually be rebuilt from what is shipped?
try
    [net, Mchk] = mdn_load_model(MDN_FILE);
    fprintf('    [ OK ] network rebuilt from stored weights (M = %d, %d layers)\n', ...
        Mchk, numel(net.Layers));
catch ME
    fprintf('    [FAIL] network could not be rebuilt: %s\n', ME.message);
    ok = false;
end
fprintf('\n');

%% [5] Normalization --------------------------------------------------
% These are global statistics, computed once over all modeled land cells
% across the calibration period, not per patch. Every patch shares them,
% so the numbers below are the same whichever patch is shipped.
fprintf('[5] normParams (global, calibration period; never recomputed here)\n');
n = B.normParams;
fprintf('    smap     mean %+9.6f  std %9.6f\n', n.smap_mean,     n.smap_std);
fprintf('    precip   mean %+9.6f  std %9.6f\n', n.precip_mean,   n.precip_std);
fprintf('    wbm      mean %+9.6f  std %9.6f\n', n.wbm_mean,      n.wbm_std);
fprintf('    residual mean %+9.6f  std %9.6f\n', n.residual_mean, n.residual_std);
ok = pos_check('smap_std',     n.smap_std)     && ok;
ok = pos_check('precip_std',   n.precip_std)   && ok;
ok = pos_check('wbm_std',      n.wbm_std)      && ok;
ok = pos_check('residual_std', n.residual_std) && ok;
fprintf('\n');

%% [6] Scored cells ---------------------------------------------------
fprintf('[6] Scored cells\n');
n_prod = numel(B.performance.kge_mdn_per_grid);
n_wbm  = sum(P.modeling_mask(:));
fprintf('    model per-cell arrays : %d\n', n_prod);
fprintf('    WBM modeled cells    : %d\n', n_wbm);
fprintf('    land cells in patch   : %d of %d\n', sum(S.land_mask(:)), numel(S.land_mask));
ok = eq_check('per-cell arrays vs WBM modeled cells', n_prod, n_wbm) && ok;
fprintf('\n');

%% Verdict ------------------------------------------------------------
fprintf('========================================\n');
if ok
    fprintf('All consistency checks passed.\n');
    fprintf('Next: runExample_quick\n');
else
    fprintf('Some checks FAILED -- do not trust a reproduction run until they are fixed.\n');
end
fprintf('========================================\n');

%% local ---------------------------------------------------------------
function ok = eq_check(label, a, b)
    ok = isequal(double(a), double(b));
    flag = '[ OK ]'; if ~ok, flag = '[FAIL]'; end
    fprintf('    %s %-42s %s vs %s\n', flag, label, mat2str(a), mat2str(b));
end

function ok = str_check(label, a, b)
    ok = strcmp(a, b);
    flag = '[ OK ]'; if ~ok, flag = '[FAIL]'; end
    fprintf('    %s %-42s %s\n', flag, label, a);
end

function ok = pos_check(label, v)
    ok = isfinite(v) && v > 0;
    if ~ok, fprintf('    [FAIL] %-42s %g is not a usable scale\n', label, v); end
end
