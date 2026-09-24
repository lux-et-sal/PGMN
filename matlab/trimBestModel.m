%% trimBestModel.m
% =====================================================================
% PGMN: build pretrained/best_model_M<M>.mat from a training run.
%
% The patch index is read from data/sample_patch.mat and the mixture size
% from the trained file itself, so this script never has to be edited when
% the shipped patch changes -- run extract_sample_patch first.
%
% This replaces what used to be a manual file copy. A straight copy no
% longer works, for two reasons:
%
%   1. The training file carries a `net` field, a saved dlnetwork object.
%      Loading it emits three warnings and yields a network whose custom
%      ConvLSTM layer has been silently replaced. Shipping that field
%      invites a user to load it and get wrong answers. It is dropped here;
%      mdn_load_model.m rebuilds the network from the stored weights
%      instead (save_method = 'learnables_direct').
%
%   2. The training file is ~1.4 MB of which a good part is bookkeeping
%      that a reproduction package has no use for.
%
% KEPT
%   M, input_size, normParams, patch_info, save_method
%   learnables_table / learnables_vals / state_table / state_vals
%       the weights, keyed by (Layer, Parameter)
%   performance.metrics, .metrics_wbm            pooled patch metrics
%   performance.*_per_grid                       the 8 per-cell arrays that
%       expected_outputs/metrics_reference.json is derived from
%   performance.AIC / num_params / NLL_* / n_*   model-selection record
%   performance.num_samples / num_valid / num_grids_analyzed
%   train_info                                   loss curves, stopping reason
%
% DROPPED
%   net                     unusable on reload, see above
%   performance.per_patch   single-patch package, so it is a copy of itself
%   performance.improvement_* / mean_uncertainty  recomputed by runExample_quick
%
% USAGE
%   >> trimBestModel          % preview: prints what would be written
%   >> trimBestModel(true)    % write pretrained/best_model_M*.mat
%
% There is no version control on the source tree, so the destination is
% never overwritten silently: an existing file is renamed to
% best_model_M<M>.mat.bak_<timestamp> first.
% =====================================================================

function trimBestModel(doWrite)

if nargin < 1 || isempty(doWrite), doWrite = false; end

%% USER PATHS ---------------------------------------------------------
BASE = 'C:/Users/Administrator/Desktop/WBM-LSTM_satellite_36km_EASE';
RDIR = fullfile(BASE, 'outputs', 'MDN-ConvLSTM', ...
                'outputs_individual_patches_QC_64x64_open_loop_TO_V1');

this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
addpath(this_dir);                     % ConvLSTMLayer must be resolvable

fprintf('========================================\n');
fprintf('PGMN: building the pretrained model file\n');
if ~doWrite
    fprintf('*** preview -- nothing is written. Run trimBestModel(true) ***\n');
end
fprintf('========================================\n\n');

% Which patch, and which M, are decided by the shipped data and by the
% training run -- not repeated here. Run extract_sample_patch first.
DATA_FILE = fullfile(repo_root, 'data', 'sample_patch.mat');
assert(exist(DATA_FILE,'file')==2, ...
    'Run extract_sample_patch first -- this script takes the patch index from it.');
Sm = load(DATA_FILE, 'meta');
PATCH_IDX = Sm.meta.patch_idx;

md = dir(fullfile(RDIR, sprintf('Patch_%04d', PATCH_IDX), 'models', 'best_model_M*.mat'));
assert(~isempty(md), 'No trained model under %s', ...
       fullfile(RDIR, sprintf('Patch_%04d', PATCH_IDX), 'models'));
assert(numel(md) == 1, ...
    ['%d candidate models under Patch_%04d/models. The training run should ' ...
     'leave exactly one; resolve it before shipping.'], numel(md), PATCH_IDX);
SOURCE = fullfile(md(1).folder, md(1).name);
DEST   = fullfile(repo_root, 'pretrained', md(1).name);

% The `net` field will warn on load; that is expected and is exactly why
% it is dropped. Suppress the noise so the real output stays readable.
ws = warning('off', 'all');
raw = load(SOURCE);
warning(ws);

assert(isfield(raw, 'save_method') && strcmp(raw.save_method, 'learnables_direct'), ...
    'Source model is not in learnables_direct format -- cannot ship it.');

out = struct();
out.M                = raw.M;
out.input_size       = raw.input_size;
out.normParams       = raw.normParams;
out.patch_info       = raw.patch_info;
out.save_method      = raw.save_method;
out.learnables_table = raw.learnables_table;
out.learnables_vals  = raw.learnables_vals;
out.state_table      = raw.state_table;
out.state_vals       = raw.state_vals;
if isfield(raw, 'train_info'), out.train_info = raw.train_info; end

perf_fields = {'metrics','metrics_wbm', ...
    'rmse_wbm_per_grid','rmse_mdn_per_grid','bias_wbm_per_grid','bias_mdn_per_grid', ...
    'r_wbm_per_grid','r_mdn_per_grid','kge_wbm_per_grid','kge_mdn_per_grid', ...
    'AIC','num_params','NLL_cali','n_cali','NLL_valid','n_valid','NLL_test','n_test', ...
    'num_samples','num_valid','num_grids_analyzed'};
P = struct();
for i = 1:numel(perf_fields)
    f = perf_fields{i};
    if isfield(raw.performance, f), P.(f) = raw.performance.(f); end
end
out.performance = P;

%% Report -------------------------------------------------------------
nlearn = 0;
for i = 1:numel(out.learnables_vals), nlearn = nlearn + numel(out.learnables_vals{i}); end
d = dir(SOURCE);

fprintf('  source            : %s (%.1f KB)\n', SOURCE, d.bytes/1024);
fprintf('  M                 : %d\n', out.M);
fprintf('  input_size        : %s\n', mat2str(out.input_size));
fprintf('  learnable values  : %d in %d tensors\n', nlearn, numel(out.learnables_vals));
fprintf('  AIC k (Weights+Bias): %d\n', P.num_params);
fprintf('  per-grid arrays   : %d cells\n', numel(P.kge_mdn_per_grid));
fprintf('  dropped fields    : %s\n', strjoin(setdiff(fieldnames(raw)', fieldnames(out)'), ', '));
fprintf('\n  destination       : %s\n', DEST);

if ~doWrite
    fprintf('\nPreview complete. Run trimBestModel(true) to write.\n');
    fprintf('========================================\n');
    return;
end

%% Write --------------------------------------------------------------
if exist(DEST, 'file') == 2
    bak = [DEST '.bak_' datestr(now, 'yyyymmdd_HHMMSS')];
    movefile(DEST, bak);
    fprintf('  existing file preserved as %s\n', bak);
end
dd = fileparts(DEST);
if ~exist(dd, 'dir'); mkdir(dd); end

save(DEST, '-struct', 'out', '-v7.3');
d2 = dir(DEST);
fprintf('  written (%.1f KB)\n', d2.bytes/1024);
fprintf('========================================\n');

end
