%% checkSystemRequirements.m
% =====================================================================
% Verify that this MATLAB environment can run runExample_quick.m.
%
% Reports on:
%   1. MATLAB version
%   2. Required toolboxes
%   3. Optional toolboxes
%   4. CPU and memory
%   5. GPU
%   6. Shipped files
%
% REFERENCE MACHINE (where the published numbers were produced)
%   OS       Windows Server 2022 Standard
%   CPU      Intel Xeon, 16+ physical cores
%   RAM      256 GB   (runExample_quick peaks near 3 GB)
%   GPU      NVIDIA RTX A5000, 24 GB
%   MATLAB   R2025b
%
% WHAT IS ACTUALLY NEEDED
%   Far less than the reference machine. PGMN ships the water balance
%   baseline already simulated, so this package only runs inference on one
%   64 x 64 patch for 995 days: about 8 seconds on CPU, 7 on GPU, in a few
%   GB of memory.
% =====================================================================

clear; clc;
fprintf('========================================\n');
fprintf('PGMN system requirement check\n');
fprintf('========================================\n\n');

GREEN  = '[ OK ]';
YELLOW = '[WARN]';
RED    = '[FAIL]';
all_ok = true;

%% [1] MATLAB version --------------------------------------------------
v = ver('MATLAB');
year = sscanf(v.Release, '(R%d');
fprintf('[1] MATLAB version: %s %s\n', v.Name, v.Release);
if year >= 2024
    fprintf('    %s R2024a or later\n', GREEN);
else
    fprintf('    %s R2024a or later required (found %s)\n', RED, v.Release);
    fprintf('           dlnetwork and the custom ConvLSTM layer rely on it.\n');
    all_ok = false;
end

%% [2] Required toolboxes ---------------------------------------------
fprintf('\n[2] Required toolboxes\n');
required = { ...
    'Deep Learning Toolbox',                  'nnet',  'dlnetwork, dlarray, predict'; ...
    'Statistics and Machine Learning Toolbox','stats', 'corr and prctile in the metric code'};
for i = 1:size(required,1)
    if have(required{i,2})
        fprintf('    %s %-42s %s\n', GREEN, required{i,1}, required{i,3});
    else
        fprintf('    %s %-42s NOT INSTALLED\n', RED, required{i,1});
        all_ok = false;
    end
end

%% [3] Optional toolboxes ---------------------------------------------
fprintf('\n[3] Optional toolboxes\n');
if have('parallel')
    fprintf('    %s %-42s enables the GPU path\n', GREEN, 'Parallel Computing Toolbox');
else
    fprintf('    %s %-42s absent: CPU only, still ~8 s\n', YELLOW, 'Parallel Computing Toolbox');
end

%% [4] CPU and memory -------------------------------------------------
fprintf('\n[4] CPU and memory\n');
n_logical = feature('numcores');
fprintf('    logical cores : %d\n', n_logical);
if n_logical >= 4
    fprintf('    %s 4 or more cores\n', GREEN);
else
    fprintf('    %s only %d cores; inference will be slower but will finish\n', ...
            YELLOW, n_logical);
end
try
    [~, sys] = memory;
    ram_gb = double(sys.PhysicalMemory.Total) / 1e9;
    fprintf('    physical RAM  : %.1f GB\n', ram_gb);
    if ram_gb >= 8
        fprintf('    %s 8 GB or more\n', GREEN);
    else
        fprintf('    %s only %.1f GB; the batch loop may swap\n', YELLOW, ram_gb);
    end
catch
    fprintf('    %s memory() is Windows-only; skipping\n', YELLOW);
end

%% [5] GPU ------------------------------------------------------------
fprintf('\n[5] GPU\n');
gpu_ok = false;
try, gpu_ok = canUseGPU(); catch, end
if gpu_ok
    g = gpuDevice;
    fprintf('    %s %s, %.1f GB, compute capability %s\n', GREEN, ...
            g.Name, double(g.TotalMemory)/1e9, g.ComputeCapability);
else
    fprintf('    %s no usable GPU; inference runs on CPU (about 8 s)\n', YELLOW);
end

%% [6] Shipped files --------------------------------------------------
fprintf('\n[6] Shipped files\n');
this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);

all_ok = need_one(fullfile(repo_root,'data','sample_patch.mat'), ...
                  repo_root, GREEN, RED) && all_ok;
all_ok = need_glob(fullfile(repo_root,'pretrained','best_model_M*.mat'), ...
                   repo_root, GREEN, RED) && all_ok;
all_ok = need_glob(fullfile(repo_root,'pretrained','wbm_params_patch*.mat'), ...
                   repo_root, GREEN, RED) && all_ok;
all_ok = need_one(fullfile(repo_root,'expected_outputs','metrics_reference.json'), ...
                  repo_root, GREEN, RED) && all_ok;

% ConvLSTMLayer must be resolvable or the network cannot be rebuilt.
if exist('ConvLSTMLayer', 'class') == 8 || exist(fullfile(this_dir,'ConvLSTMLayer.m'),'file') == 2
    fprintf('    %s matlab/ConvLSTMLayer.m\n', GREEN);
else
    fprintf('    %s matlab/ConvLSTMLayer.m MISSING -- the model cannot be loaded\n', RED);
    all_ok = false;
end

%% Summary ------------------------------------------------------------
fprintf('\n========================================\n');
if all_ok
    fprintf('All required checks passed. Next:\n');
    fprintf('  >> verify_loaded_files      %% files agree with each other\n');
    fprintf('  >> runExample_quick         %% reproduce\n');
else
    fprintf('Some required checks FAILED. See the messages above.\n');
    fprintf('For missing data or model files, see SETUP_NOTES.md.\n');
end
fprintf('========================================\n');

%% local --------------------------------------------------------------
function tf = have(short_name)
    tf = ~isempty(ver(short_name));
end

function ok = need_one(f, root, GREEN, RED)
    ok = exist(f, 'file') == 2;
    if ok
        d = dir(f);
        fprintf('    %s %-46s %s\n', GREEN, rel(f, root), human(d.bytes));
    else
        fprintf('    %s %-46s MISSING -- see SETUP_NOTES.md\n', RED, rel(f, root));
    end
end

function ok = need_glob(pattern, root, GREEN, RED)
    d = dir(pattern);
    ok = numel(d) == 1;
    if ok
        f = fullfile(d(1).folder, d(1).name);
        fprintf('    %s %-46s %s\n', GREEN, rel(f, root), human(d(1).bytes));
    elseif isempty(d)
        fprintf('    %s %-46s MISSING -- see SETUP_NOTES.md\n', RED, rel(pattern, root));
    else
        fprintf('    %s %-46s %d matches, expected exactly one\n', ...
                RED, rel(pattern, root), numel(d));
    end
end

function r = rel(f, root)
    r = strrep(strrep(f, [root filesep], ''), '\', '/');
end

function s = human(bytes)
    if bytes >= 1e6
        s = sprintf('%.1f MB', bytes/1e6);
    else
        s = sprintf('%.0f KB', bytes/1e3);
    end
end
