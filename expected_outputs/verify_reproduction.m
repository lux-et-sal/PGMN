%% verify_reproduction.m
% =====================================================================
% Compare what runExample_quick.m produced (outputs/metrics_latest.mat)
% against metrics_reference.json.
%
% Three groups are checked:
%   [1] domain      the run scored the cells and days it was supposed to
%   [2] accuracy    WBM baseline and PGMN reconstruction, 4 metrics each
%   [3] uncertainty coverage, sharpness and the distributional score
%
% Group [1] is checked first and on purpose. A run that silently scored a
% different set of cells can still land inside the metric tolerances, and
% then the PASS would mean nothing.
%
% Every reference number is a median across scored cells. Tolerances are
% absolute and per metric; they are set from measured GPU-versus-CPU spread
% (see make_reference.m), not chosen for comfort.
%
% PASS : every checked quantity is within tolerance.
% FAIL : at least one is not, or outputs/metrics_latest.mat is missing.
%
% Run from the expected_outputs/ directory:
%   >> verify_reproduction
% =====================================================================

clear; clc;

this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);

LATEST_FILE = fullfile(repo_root, 'outputs', 'metrics_latest.mat');
REF_FILE    = fullfile(this_dir,  'metrics_reference.json');

assert(exist(LATEST_FILE,'file')==2, ...
       'Run matlab/runExample_quick.m first; %s not found.', LATEST_FILE);
assert(exist(REF_FILE,'file')==2, ...
       'Reference file %s missing.', REF_FILE);

L   = load(LATEST_FILE);
ref = jsondecode(fileread(REF_FILE));
tol = ref.tolerance;

fprintf('========================================\n');
fprintf('PGMN reproduction check\n');
fprintf('========================================\n');
fprintf('patch %d, %s\n', ref.patch_idx, ref.grid);
fprintf('evaluation %s to %s (%d of %d block days; %d warm-up)\n', ...
    ref.evaluation_period.start, end_field(ref), ...
    ref.evaluation_period.scored_days, ref.evaluation_period.block_days, ...
    ref.evaluation_period.warmup_days);
fprintf('convention: %s\n\n', ref.convention);

ok = true;

%% [1] Domain ---------------------------------------------------------
fprintf('--- [1] Domain ---\n');
ok = check_int('scored cells', ref.n_grid, L.metrics_mdn.n_grid) && ok;
ok = check_int('scored cells (uncertainty)', ref.n_grid, L.unc.n_grid) && ok;
ok = check_int('scored cells (WBM baseline)', ref.n_grid, L.metrics_wbm.n_grid) && ok;
fprintf('\n');

%% [2] Accuracy -------------------------------------------------------
fprintf('--- [2] Accuracy (grid-wise median) ---\n');
keys_acc = {'KGE','RMSE','R','Bias'};
ok = compare_block('WBM baseline',        ref.accuracy.wbm_baseline,        L.metrics_wbm, keys_acc, tol) && ok;
ok = compare_block('PGMN reconstruction', ref.accuracy.pgmn_reconstruction, L.metrics_mdn, keys_acc, tol) && ok;

%% [3] Uncertainty ----------------------------------------------------
fprintf('--- [3] Predictive distribution ---\n');
keys_unc = {'PICP95','q','MPIW','MAE','CRPS','MA','SB','CRPS_reduction_pct'};
ok = compare_block('calibration', ref.uncertainty, L.unc, keys_unc, tol) && ok;

%% Verdict ------------------------------------------------------------
fprintf('========================================\n');
if ok
    fprintf('REPRODUCTION SUCCESS -- every checked quantity within tolerance.\n');
else
    fprintf('REPRODUCTION FAILED  -- see the flagged rows above.\n');
end
fprintf('========================================\n');

%% local ---------------------------------------------------------------
function s = end_field(ref)
    % 'end' is a keyword, so jsondecode's field of that name needs indirection.
    f = ref.evaluation_period;
    s = f.('end');
end

function ok = check_int(label, expected, got)
    ok = isequal(double(expected), double(got));
    flag = '[ OK ]';  if ~ok, flag = '[FAIL]'; end
    fprintf('  %s %-28s ref=%d  got=%d\n', flag, label, expected, got);
end

function ok = compare_block(label, ref, got, keys, tol)
    fprintf('  %s\n', label);
    ok = true;
    for i = 1:numel(keys)
        k = keys{i};
        if ~isfield(ref, k) || ~isfield(got, k)
            fprintf('    [FAIL] %-20s missing from %s\n', k, ...
                    ternary(~isfield(ref,k), 'reference', 'run'));
            ok = false;  continue;
        end
        r = double(ref.(k));
        g = double(got.(k));
        t = double(tol.(k));
        d = g - r;
        within = abs(d) <= t;
        flag = '[ OK ]';  if ~within, flag = '[FAIL]';  ok = false; end
        fprintf('    %s %-20s ref=%+11.6f  got=%+11.6f  delta=%+9.2e  tol=%.0e\n', ...
                flag, k, r, g, d, t);
    end
    fprintf('\n');
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end
