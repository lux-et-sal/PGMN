%% plotReproduction.m
% =====================================================================
% PGMN: the representative reproducibility figure, 2 x 2.
%
%   (a) Soil moisture over the evaluation period at the cell holding an
%       ISMN station used in the paper's in-situ comparison: SMAP
%       observation, WBM baseline, PGMN reconstruction with a +/- 1 sigma
%       band. One real cell, not a
%       domain average -- an average would smooth away the day-to-day
%       behavior the panel exists to show, and would correspond to no
%       actual place. If the shipped patch has no such station the panel
%       falls back to the median across scored cells.
%
%   (b) Per-cell KGE improvement, KGE_PGMN - KGE_WBM, with the cell from
%       (a) marked. This is where the residual correction earns its place:
%       the WBM baseline it corrects is the same field in both terms, so
%       the difference isolates what the network added.
%
%   (c) Reliability diagram. Nominal confidence on the x axis, realized
%       coverage on the y. If the predictive distribution is honest the
%       curve sits on the diagonal at every level, not just at 0.95. The
%       band is the interquartile range across cells; the single PICP_95
%       point is marked to show how little of the curve it pins down.
%
%   (d) Reproduction margin. For every quantity verify_reproduction.m
%       checks, the bar is |computed - reference| divided by that metric's
%       tolerance. Anything reaching 1 would fail. Plotting the ratio
%       rather than the raw delta lets metrics on different scales share
%       one axis.
%
% Inputs (produced by runExample_quick.m, plus the shipped files)
%   data/sample_patch.mat
%   outputs/metrics_latest.mat
%   outputs/reconstruction.mat
%   expected_outputs/metrics_reference.json
%
% Outputs
%   outputs/reproduction_figure.png  (300 dpi)
%   outputs/reproduction_figure.fig
% =====================================================================

clear; clc; close all;

%% Paths
this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
addpath(this_dir);

DATA_FILE = fullfile(repo_root, 'data',            'sample_patch.mat');
LATEST    = fullfile(repo_root, 'outputs',         'metrics_latest.mat');
RECON     = fullfile(repo_root, 'outputs',         'reconstruction.mat');
REF_FILE  = fullfile(repo_root, 'expected_outputs','metrics_reference.json');
FIG_PNG   = fullfile(repo_root, 'outputs',         'reproduction_figure.png');
FIG_FIG   = fullfile(repo_root, 'outputs',         'reproduction_figure.fig');

assert(exist(LATEST,'file')==2 && exist(RECON,'file')==2, ...
    'outputs/ is incomplete -- run runExample_quick first.');

S   = load(DATA_FILE, 'LAT', 'LON', 'meta', 'station');
L   = load(LATEST);
R   = load(RECON);
ref = jsondecode(fileread(REF_FILE));

FS = 11;
set(groot, 'DefaultAxesFontSize', FS);

fig = figure('Units','inches','Position',[0.5 0.5 13 8.4], ...
             'Color','w','PaperPositionMode','auto');
tl = tiledlayout(fig, 2, 2, 'TileSpacing','compact', 'Padding','compact');

% Cells that were scored, as a mask on the 64 x 64 patch
[H, W, ~] = size(R.smap_true);
scored = false(H, W);
scored(sub2ind([H W], L.metrics_mdn.row, L.metrics_mdn.col)) = true;

%% ----- (a) time series ----------------------------------------------
% Prefer the cell that carries an in-situ reference station. A
% domain median smooths away exactly the day-to-day behavior a reader
% wants to inspect, and it corresponds to no real place. Fall back to the
% median only if the shipped patch contains no reference station.
ax1 = nexttile(tl); hold(ax1,'on'); box(ax1,'on');

t = R.time(:);
use_station = isfield(S,'station') && isstruct(S.station) && S.station.present ...
              && scored(S.station.row, S.station.col);

if use_station
    ri = S.station.row;  ci = S.station.col;
    obs = squeeze(R.smap_true   (ri, ci, :));
    wbm = squeeze(R.wbm         (ri, ci, :));
    mdn = squeeze(R.smap_pred   (ri, ci, :));
    sig = squeeze(R.residual_std(ri, ci, :));
    obs(~squeeze(R.mask(ri, ci, :))) = NaN;
    j = find(L.metrics_mdn.row == ri & L.metrics_mdn.col == ci, 1);
    % The KGE quoted here is against SMAP at this cell, over the full
    % evaluation window. It is not the station-level KGE the in-situ
    % validation reports, which scores against the probe itself.
    ttl = sprintf('%s / %s  (%.3f\\circN, %.3f\\circE)   KGE vs SMAP  %.3f \\leftarrow %.3f', ...
        S.station.network, strrep(char(S.station.name), '_', ' '), ...
        S.station.cell_lat, S.station.cell_lon, ...
        L.metrics_mdn.per_grid.KGE(j), L.metrics_wbm.per_grid.KGE(j));
else
    obs = cell_median(R.smap_true,    scored);
    wbm = cell_median(R.wbm,          scored);
    mdn = cell_median(R.smap_pred,    scored);
    sig = cell_median(R.residual_std, scored);
    ttl = 'Median over scored cells, evaluation period';
end

good = ~isnan(mdn) & ~isnan(sig);
fill(ax1, [t(good); flipud(t(good))], ...
         [mdn(good)+sig(good); flipud(mdn(good)-sig(good))], ...
         [0.82 0.88 0.96], 'EdgeColor','none', 'FaceAlpha',0.8, ...
         'DisplayName','PGMN \pm 1\sigma');
plot(ax1, t, wbm, '-', 'LineWidth',1.1, 'Color',[0.85 0.33 0.10], 'DisplayName','WBM');
plot(ax1, t, mdn, '-', 'LineWidth',1.3, 'Color',[0.10 0.35 0.75], 'DisplayName','PGMN');
plot(ax1, t, obs, 'o', 'MarkerSize',2.4, 'MarkerFaceColor',[0.15 0.15 0.15], ...
     'MarkerEdgeColor','none', 'DisplayName','SMAP');

ylabel(ax1, '\theta  (m^3 m^{-3})');
title(ax1, ttl, 'FontWeight','normal');
legend(ax1, 'Location','north', 'FontSize',9, 'Box','off', 'NumColumns',4);
xlim(ax1, [t(1) t(end)]);  xtickformat(ax1, 'yyyy-MM');
yl = ylim(ax1);  ylim(ax1, [yl(1), yl(1) + (yl(2)-yl(1))*1.18]);
panel_label(ax1, '(a)');

%% ----- (b) delta KGE map --------------------------------------------
ax2 = nexttile(tl);

dk = nan(H, W);
dk(sub2ind([H W], L.metrics_mdn.row, L.metrics_mdn.col)) = ...
    L.metrics_mdn.per_grid.KGE - L.metrics_wbm.per_grid.KGE;

imagesc(ax2, S.LON(1,:), S.LAT(:,1), dk, 'AlphaData', ~isnan(dk));
set(ax2, 'YDir','normal', 'Color',[0.94 0.94 0.94]);
axis(ax2, 'image');

% Symmetric limits so that zero stays white, but set from the 95th
% percentile rather than the extreme. A handful of cells where the WBM
% collapses give |dKGE| near 2; scaling to those would flatten the rest.
lim = max(prctile(abs(dk(~isnan(dk))), 95), 1e-3);
clim(ax2, [-lim lim]);
colormap(ax2, diverging_bwr(256));
cb = colorbar(ax2, 'eastoutside');
cb.Label.String = '\DeltaKGE = KGE_{PGMN} - KGE_{WBM}';
n_clip = sum(abs(dk(:)) > lim);
xlabel(ax2, 'Longitude (\circE)');  ylabel(ax2, 'Latitude (\circN)');
title(ax2, sprintf('Per-cell KGE improvement (n = %d, %d beyond \\pm%.2f)', ...
      L.metrics_mdn.n_grid, n_clip, lim), 'FontWeight','normal');

% Zoom to the cells that carry data; the rest of the 64 x 64 box is ocean.
[rr, cc] = find(~isnan(dk));
pad = 2;
rsel = max(min(rr)-pad,1) : min(max(rr)+pad, H);
csel = max(min(cc)-pad,1) : min(max(cc)+pad, W);
% Row index runs north to south on an EASE grid, so sort rather than assume.
xlim(ax2, sort([S.LON(1, csel(1)), S.LON(1, csel(end))]));
ylim(ax2, sort([S.LAT(rsel(1), 1), S.LAT(rsel(end), 1)]));

% Mark the cell shown in (a) so the two panels are visibly the same place.
if use_station
    hold(ax2, 'on');
    plot(ax2, S.station.cell_lon, S.station.cell_lat, 'p', ...
        'MarkerSize',14, 'MarkerFaceColor',[0.95 0.72 0.20], ...
        'MarkerEdgeColor',[0.15 0.15 0.15], 'LineWidth',0.8);
    text(ax2, S.station.cell_lon, S.station.cell_lat, ...
        ['  ' strrep(char(S.station.name), '_', ' ')], ...
        'FontSize',9, 'VerticalAlignment','middle', 'HorizontalAlignment','left');
end
panel_label(ax2, '(b)');

%% ----- (c) reliability diagram --------------------------------------
ax3 = nexttile(tl); hold(ax3,'on'); box(ax3,'on');

c   = L.unc.confidence_levels(:);
PC  = L.unc.coverage_curve;                       % [n_cell, K]
lo  = prctile(PC, 25, 1)';   hi = prctile(PC, 75, 1)';   md = median(PC, 1)';

fill(ax3, [c; flipud(c)], [hi; flipud(lo)], [0.82 0.88 0.96], ...
     'EdgeColor','none', 'FaceAlpha',0.85, 'DisplayName','IQR across cells');
plot(ax3, [0 1], [0 1], '--', 'Color',[0.45 0.45 0.45], 'LineWidth',1, ...
     'DisplayName','perfect calibration');
plot(ax3, c, md, '-o', 'Color',[0.10 0.35 0.75], 'LineWidth',1.4, ...
     'MarkerSize',3.5, 'MarkerFaceColor',[0.10 0.35 0.75], ...
     'DisplayName','median across cells');
plot(ax3, 0.95, L.unc.PICP95, 'p', 'MarkerSize',13, ...
     'MarkerFaceColor',[0.95 0.72 0.20], 'MarkerEdgeColor',[0.4 0.3 0.05], ...
     'DisplayName', sprintf('PICP_{95} = %.3f', L.unc.PICP95));

xlabel(ax3, 'Nominal confidence level');
ylabel(ax3, 'Realized coverage');
title(ax3, sprintf('Reliability   (MA = %.3f, SB = %+.3f)', L.unc.MA, L.unc.SB), ...
      'FontWeight','normal');
legend(ax3, 'Location','southeast', 'FontSize',9, 'Box','off');
axis(ax3, [0 1 0 1]);  axis(ax3, 'square');
panel_label(ax3, '(c)');

%% ----- (d) reproduction margin --------------------------------------
ax4 = nexttile(tl); hold(ax4,'on'); box(ax4,'on');

rows = [ ...
    build('KGE',  'WBM',  ref.accuracy.wbm_baseline,        L.metrics_wbm, ref.tolerance)
    build('RMSE', 'WBM',  ref.accuracy.wbm_baseline,        L.metrics_wbm, ref.tolerance)
    build('R',    'WBM',  ref.accuracy.wbm_baseline,        L.metrics_wbm, ref.tolerance)
    build('Bias', 'WBM',  ref.accuracy.wbm_baseline,        L.metrics_wbm, ref.tolerance)
    build('KGE',  'PGMN', ref.accuracy.pgmn_reconstruction, L.metrics_mdn, ref.tolerance)
    build('RMSE', 'PGMN', ref.accuracy.pgmn_reconstruction, L.metrics_mdn, ref.tolerance)
    build('R',    'PGMN', ref.accuracy.pgmn_reconstruction, L.metrics_mdn, ref.tolerance)
    build('Bias', 'PGMN', ref.accuracy.pgmn_reconstruction, L.metrics_mdn, ref.tolerance)
    build('PICP95','unc', ref.uncertainty, L.unc, ref.tolerance)
    build('q',     'unc', ref.uncertainty, L.unc, ref.tolerance)
    build('MPIW',  'unc', ref.uncertainty, L.unc, ref.tolerance)
    build('MAE',   'unc', ref.uncertainty, L.unc, ref.tolerance)
    build('CRPS',  'unc', ref.uncertainty, L.unc, ref.tolerance)
    build('MA',    'unc', ref.uncertainty, L.unc, ref.tolerance)
    build('SB',    'unc', ref.uncertainty, L.unc, ref.tolerance)
    build('CRPS_reduction_pct','unc', ref.uncertainty, L.unc, ref.tolerance)];

margin = [rows.margin];
labels = {rows.label};
x = 1:numel(margin);

bar(ax4, x, max(margin, 1e-6), 0.65, 'FaceColor',[0.35 0.55 0.80], 'EdgeColor','none');
yline(ax4, 1, '-', 'FAIL threshold', 'Color',[0.80 0.20 0.15], 'LineWidth',1.3, ...
      'LabelHorizontalAlignment','right', 'LabelVerticalAlignment','bottom', 'FontSize',9);
set(ax4, 'YScale','log', 'XTick',x, 'XTickLabel',labels, 'XTickLabelRotation',55, ...
         'TickLabelInterpreter','none');
ylim(ax4, [1e-6 3]);
ylabel(ax4, '|computed - reference| / tolerance');
title(ax4, sprintf('Reproduction margin (max %.1e of tolerance)', max(margin)), ...
      'FontWeight','normal');
panel_label(ax4, '(d)');

%% Save
print(fig, FIG_PNG, '-dpng', '-r300');
savefig(fig, FIG_FIG);
fprintf('Saved %s\n', FIG_PNG);
fprintf('Saved %s\n', FIG_FIG);

%% =====================================================================
%% local helpers
%% =====================================================================
function panel_label(ax, lbl)
    text(ax, 0.015, 0.975, lbl, 'Units','normalized', ...
         'FontWeight','bold', 'FontSize',13, 'VerticalAlignment','top');
end

function v = cell_median(A, mask2d)
% Median over the scored cells at each time step. Cells with no observation
% at that step drop out, so the SMAP series is sparse by construction.
    T = size(A, 3);
    v = nan(T, 1);
    for k = 1:T
        s = A(:,:,k);
        sel = mask2d & ~isnan(s);
        if any(sel(:)); v(k) = median(s(sel)); end
    end
end

function r = build(key, group, refblk, gotblk, tol)
    r.label  = [group ' ' key];
    if strcmp(group, 'unc'), r.label = key; end
    r.margin = abs(double(gotblk.(key)) - double(refblk.(key))) / double(tol.(key));
end

function cmap = diverging_bwr(n)
    half = floor(n/2);
    blue = [linspace(0.10,1,half)' linspace(0.35,1,half)' linspace(0.75,1,half)'];
    red  = [linspace(1,0.85,n-half)' linspace(1,0.30,n-half)' linspace(1,0.20,n-half)'];
    cmap = [blue; red];
end
