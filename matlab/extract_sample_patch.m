%% extract_sample_patch.m
% =====================================================================
% PGMN: build data/sample_patch.mat for the shipped sample patch.
%
% WHAT THIS SHIPS AND WHY
%   The network never sees raw SMAP / MSWEP / WBM arrays. It sees one
%   quantized HDF5 patch file produced by the authors' preprocessing chain
%   (not part of this repository).
%   Every channel is normalized and then stored as a scaled integer with
%   scale_factor = 1000 and fill_value = -32768. The normalization
%   statistics are global, not per patch: they are computed once over all
%   modeled land cells across the calibration period, so every patch
%   shares one set. That is why normParams in the model file is identical
%   for any patch you might ship.
%
%   So the honest thing to ship is that file's Evaluation block, verbatim
%   and still quantized. Re-deriving it from the source archives would
%   re-introduce the exact rounding the model was trained against and the
%   reproduction would drift in the fourth decimal for no benefit.
%
%   Consequence: PGMN reproduces the Evaluation-period numbers bit for bit,
%   and does NOT re-run the water balance model. The WBM baseline is
%   already inside channel 2 (see below). wbm_*.m are kept in matlab/ as
%   documentation of the physics, not as part of the run path.
%
% WHAT IS IN THE OUTPUT FILE
%   XVal        int32  [64 64 3 1004]  normalized inputs, quantized
%                       ch1 = precipitation(t)
%                       ch2 = WBM soil moisture(t)      <- the physics baseline
%                       ch3 = residual(t-1) = WBM - SMAP
%   ResAvailVal uint8  [64 64 1004]    ch4: 1 if SMAP was observed at t-1
%   YVal_Direct int16  [64 64 1004]    normalized SMAP(t), the target
%   MaskVal     logical[64 64 1004]    1 where SMAP(t) exists
%   scale_factor / fill_value          dequantisation constants
%   land_mask   logical[64 64]         EASE M36 land cells inside the patch
%
%   The integer widths (int32 for X, int16 for Y) are inherited from the
%   HDF5 file as written; they are preserved rather than normalized so the
%   shipped arrays are byte-comparable with the source.
%   LAT LON     single [64 64]         EASE-Grid 2.0 M36 cell centers
%   time        datetime[1004 1]       physical date of each Evaluation step
%   split                              index bookkeeping (see below)
%   station                            the in-situ reference cell, if the
%                                      patch contains one (see [3b])
%   meta                               provenance, basenames only
%
%   Physical units are recovered with the normParams stored in the shipped
%   model file, pretrained/best_model_M<M>.mat:
%       precip = XVal(:,:,1,:)*precip_std + precip_mean      [mm/day]
%       wbm    = XVal(:,:,2,:)*wbm_std    + wbm_mean         [m^3/m^3]
%       smap   = YVal_Direct  *smap_std   + smap_mean        [m^3/m^3]
%   mdn_reconstruct.m does exactly this, matching the production evaluation.
%
% USER PATHS
%   Edit the block below to match your machine before running.
% =====================================================================

clear; clc;

%% USER PATHS ---------------------------------------------------------
BASE     = 'C:/Users/Administrator/Desktop/WBM-LSTM_satellite_36km_EASE';
PATCH_IDX = 20;          % central United States; see README section 6

H5_DIR   = 'D:/patch_36km_EASE/Unified_Patches_QC_2022_TO/patches_unified_h5_64x64';
H5_FILE  = fullfile(H5_DIR, sprintf('patch_%04d.h5', PATCH_IDX));
META_FILE= fullfile(H5_DIR, 'unified_metadata.mat');
GRID_FILE= fullfile(BASE, 'inputs', 'grid_M36.mat');

% ISMN stations used in the paper's in-situ comparison. Whichever of them
% falls inside the patch is recorded so that the figure can single that
% cell out instead of showing an anonymous domain average.
STATION_CSV = fullfile(BASE, 'outputs', 'ISMN_era5_R30', 'timeseries', ...
                       'representatives_2024.csv');

this_dir = fileparts(mfilename('fullpath'));
OUT_FILE = fullfile(fileparts(this_dir), 'data', 'sample_patch.mat');

%% --------------------------------------------------------------------
fprintf('========================================\n');
fprintf('PGMN: extracting data/sample_patch.mat\n');
fprintf('========================================\n\n');

assert(exist(H5_FILE,  'file')==2, 'patch H5 not found: %s',  H5_FILE);
assert(exist(META_FILE,'file')==2, 'metadata not found: %s',  META_FILE);
assert(exist(GRID_FILE,'file')==2, 'grid not found: %s',      GRID_FILE);

outdir = fileparts(OUT_FILE);
if ~exist(outdir, 'dir'); mkdir(outdir); end

%% [1] Patch geometry from the H5 itself ------------------------------
row_start = double(h5read(H5_FILE, '/row_start'));
row_end   = double(h5read(H5_FILE, '/row_end'));
col_start = double(h5read(H5_FILE, '/col_start'));
col_end   = double(h5read(H5_FILE, '/col_end'));
h5_idx    = double(h5read(H5_FILE, '/patch_idx'));
scale_factor = double(h5read(H5_FILE, '/scale_factor'));
fill_value   = double(h5read(H5_FILE, '/fill_value'));

assert(h5_idx == PATCH_IDX, 'H5 patch_idx=%d but PATCH_IDX=%d', h5_idx, PATCH_IDX);

fprintf('[1] Patch %d : rows %d:%d, cols %d:%d  (%d x %d)\n', ...
    PATCH_IDX, row_start, row_end, col_start, col_end, ...
    row_end-row_start+1, col_end-col_start+1);
fprintf('    quantization: scale_factor = %d, fill_value = %d\n\n', ...
    scale_factor, fill_value);

%% [2] Evaluation block, verbatim -------------------------------------
% The HDF5 dataset names are historical. '/XVal' is the EVALUATION block
% (2023-01-01 ~ 2025-09-30), not the early-stopping valid split.
fprintf('[2] Reading Evaluation block (/XVal, /ResAvailVal, ...)\n');
XVal        = h5read(H5_FILE, '/XVal');            % int32 [64 64 3 1004]
ResAvailVal = h5read(H5_FILE, '/ResAvailVal');     % uint8 [64 64 1004]
YVal_Direct = h5read(H5_FILE, '/YVal_Direct');     % int16 [64 64 1004]
MaskVal     = logical(h5read(H5_FILE, '/MaskVal'));

[Hp, Wp, Cx, Tval] = size(XVal);
fprintf('    XVal        %s  %s\n', mat2str(size(XVal)),        class(XVal));
fprintf('    ResAvailVal %s  %s\n', mat2str(size(ResAvailVal)), class(ResAvailVal));
fprintf('    YVal_Direct %s  %s\n', mat2str(size(YVal_Direct)), class(YVal_Direct));
fprintf('    MaskVal     %s  observed fraction %.3f\n\n', ...
    mat2str(size(MaskVal)), mean(MaskVal(:)));
assert(Cx == 3, 'XVal must carry 3 stored channels (ResAvail is the 4th, kept separately)');

%% [3] Coordinates ----------------------------------------------------
fprintf('[3] Coordinates (EASE-Grid 2.0 M36)\n');
G   = load(GRID_FILE, 'LAT', 'LON', 'land_mask');
LAT = single(G.LAT(row_start:row_end, col_start:col_end));
LON = single(G.LON(row_start:row_end, col_start:col_end));
land_mask = logical(G.land_mask(row_start:row_end, col_start:col_end));
fprintf('    LAT %.3f ~ %.3f, LON %.3f ~ %.3f\n', ...
    min(LAT(:)), max(LAT(:)), min(LON(:)), max(LON(:)));
fprintf('    land cells in patch: %d / %d\n\n', sum(land_mask(:)), numel(land_mask));

%% [3b] In-situ reference cell ----------------------------------------
% Locate the ISMN representative station that falls inside this patch and
% record which cell holds it. Longitude differences are scaled by cos(lat)
% so that "nearest" means nearest on the ground, not in degrees.
station = struct('present', false);
if exist(STATION_CSV, 'file') == 2
    Tst = readtable(STATION_CSV);
    for s = 1:height(Tst)
        dd = (LAT - Tst.Lat(s)).^2 + ((LON - Tst.Lon(s)) .* cosd(Tst.Lat(s))).^2;
        [dmin, k] = min(dd(:));
        [rr, cc]  = ind2sub(size(dd), k);
        % Accept only if the station really sits in this patch, not merely
        % nearest to one of its edges. A 36 km cell is about 0.32 deg, so a
        % nearest-center distance under 0.25 deg means the station is inside
        % or on the cell; the interior test rules out the border.
        if sqrt(dmin) < 0.25 && rr > 1 && rr < Hp && cc > 1 && cc < Wp
            station = struct('present', true, ...
                'network', string(Tst.Network(s)), 'name', string(Tst.Station(s)), ...
                'lat', Tst.Lat(s), 'lon', Tst.Lon(s), ...
                'row', rr, 'col', cc, ...
                'cell_lat', double(LAT(rr,cc)), 'cell_lon', double(LON(rr,cc)), ...
                'cluster', string(Tst.Group(s)), ...
                'source', 'ISMN representative station, representatives_2024.csv');
            break;
        end
    end
end
if station.present
    fprintf('[3b] In-situ reference: %s/%s (%.4f N, %.4f E)\n', ...
        station.network, station.name, station.lat, station.lon);
    fprintf('     falls in patch cell (%d, %d) centered %.4f N, %.4f E\n\n', ...
        station.row, station.col, station.cell_lat, station.cell_lon);
else
    fprintf('[3b] No ISMN representative station inside this patch.\n\n');
end

%% [4] Time axis and the three-way split ------------------------------
% Calibration block = /XTrain, 2830 steps.  Evaluation block = /XVal, 1004.
% Within Evaluation, XVal index i targets physical time
%       t = train_end_idx + i
% and a sequence needs seq_len history, so the first SCORED step is
% i = seq_len. The nine steps before it are consumed as ConvLSTM warm-up
% and are NOT part of the reported metrics.
fprintf('[4] Time axis and split\n');
Umeta = load(META_FILE, 'metadata');
Time_All = Umeta.metadata.Time_All(:);
T_total  = numel(Time_All);

train_end_idx = T_total - Tval;                    % 3835 - 1004 = 2831
time = Time_All(train_end_idx + (1:Tval));         % physical date of XVal(i)

seq_len   = 10;
scored_i  = seq_len:Tval;                          % XVal indices actually scored
train_frac = 0.80;
T_cal      = train_end_idx - 1;                    % /XTrain length = 2830
T_fit      = floor(T_cal * train_frac);

split = struct( ...
    'seq_len',           seq_len, ...
    'train_frac',        train_frac, ...
    'train_end_idx',     train_end_idx, ...
    'T_total',           T_total, ...
    'T_calibration',     T_cal, ...
    'n_train',           numel(seq_len:T_fit), ...
    'n_valid',           numel((T_fit+seq_len):T_cal), ...
    'T_evaluation',      Tval, ...
    'n_evaluation',      numel(scored_i), ...
    'scored_first_idx',  scored_i(1), ...
    'scored_first_date', time(scored_i(1)), ...
    'scored_last_date',  time(end), ...
    'note', ['XVal index i -> physical time train_end_idx+i. ' ...
             'Only i >= seq_len is scored; earlier steps are warm-up.']);

fprintf('    full record      : %s ~ %s  (%d days)\n', ...
    datestr(Time_All(1)), datestr(Time_All(end)), T_total);
fprintf('    Calibration      : %d steps  (train %d + valid %d)\n', ...
    T_cal, split.n_train, split.n_valid);
fprintf('    Evaluation block : %d steps  (%s ~ %s)\n', ...
    Tval, datestr(time(1)), datestr(time(end)));
fprintf('    Evaluation scored: %d steps  (%s ~ %s)\n\n', ...
    split.n_evaluation, datestr(split.scored_first_date), datestr(split.scored_last_date));

%% [5] Provenance -----------------------------------------------------
meta = struct( ...
    'model',          'PGMN', ...
    'patch_idx',      PATCH_IDX, ...
    'row_range',      [row_start, row_end], ...
    'col_range',      [col_start, col_end], ...
    'lat_range',      [min(LAT,[],'all'), max(LAT,[],'all')], ...
    'lon_range',      [min(LON,[],'all'), max(LON,[],'all')], ...
    'grid',           'EASE-Grid 2.0 Global M36 (36 km equal-area), 406 x 964', ...
    'block',          'Evaluation only (HDF5 /XVal); Calibration block not shipped', ...
    'channels',       'ch1 precip(t), ch2 WBM(t), ch3 residual(t-1), ch4 ResAvail(t-1)', ...
    'quantization',   sprintf('XVal int32, YVal_Direct int16; scale_factor=%d, fill_value=%d', scale_factor, fill_value), ...
    'source_h5',      basename(H5_FILE), ...
    'source_meta',    basename(META_FILE), ...
    'source_grid',    basename(GRID_FILE), ...
    'created_at',     datestr(now, 'yyyy-mm-dd HH:MM:SS'));

save(OUT_FILE, 'XVal', 'ResAvailVal', 'YVal_Direct', 'MaskVal', ...
     'scale_factor', 'fill_value', 'LAT', 'LON', 'land_mask', ...
     'time', 'split', 'station', 'meta', '-v7.3');

f = dir(OUT_FILE);
fprintf('[5] Saved %s  (%.1f MB)\n', OUT_FILE, f.bytes/1e6);
fprintf('========================================\n');

%% Local ---------------------------------------------------------------
function s = basename(p)
    [~, n, e] = fileparts(p);
    s = [n, e];
end
