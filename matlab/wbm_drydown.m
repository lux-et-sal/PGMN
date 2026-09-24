function [paras_dd, lb, sm_outputs] = wbm_drydown(tt, pp, sm, ...
                                                   consec, pp_min, phi, hour_dif)
% wbm_drydown  Extract valid dry-down segments from a soil moisture time series.
%
% NOT ON THE RUN PATH. Documentation of the physics behind channel 2 of
% data/sample_patch.mat; runExample_quick.m ships that baseline rather than
% recomputing it. Port of WBM/S2_smdd.m. See wbm_simulate.m for how the
% production run differs.
%
%   [paras_dd, lb, sm_outputs] = wbm_drydown(tt, pp, sm, consec, pp_min, ...
%                                            phi, hour_dif)
%
% A dry-down segment is a sequence of consecutive soil moisture observations
% during which (a) precipitation is negligible, (b) soil moisture monotonically
% decreases, (c) the gap between successive observations does not exceed a
% threshold, and (d) the soil moisture stays below porosity. The procedure
% follows McColl et al. (2017) and Akbar et al. (2018) and is the same as that
% used to fit the WBM in the accompanying paper.
%
% Inputs
%   tt        datetime vector, [n,1]  (any monotonic time grid)
%   pp        precipitation,  [n,1]   (any consistent units; only the
%                                      thresholded sum between observations
%                                      is used)
%   sm        soil moisture,  [n,1]   (m^3/m^3, NaN allowed)
%   consec    minimum number of soil moisture observations required for one
%             valid dry-down segment (scalar integer, e.g. 3)
%   pp_min    threshold below which inter-observation precipitation is
%             treated as zero (scalar, mm)
%   phi       soil porosity (scalar, m^3/m^3)
%   hour_dif  maximum allowed gap between two successive soil moisture
%             observations within a dry-down segment (scalar, hours, e.g. 96)
%
% Outputs
%   paras_dd   [number_of_segments, p1, p2]   1x3
%              p1 = minimum dry-down soil moisture across all segments
%              p2 = maximum dry-down soil moisture across all segments
%   lb         per-limb table sorted by initial soil moisture, [m,5]
%              cols: [segment_id, sm_initial, dsm, dt_days, loss_rate]
%              loss_rate is in m^3/m^3/day
%   sm_outputs flagged time series, [n,3]
%              col 1: original soil moisture
%              col 2: category flag
%                     0 = NaN, 1 = noise, 2 = non-dry-down,
%                     3 = isolated single-limb (insufficient length),
%                     4 = valid dry-down observation
%              col 3: dry-down segment index (1..number_of_segments) for
%                     observations with category flag 4

    %% --- preallocate outputs ---
    sm_outputs       = nan(size(sm, 1), 3);
    sm_outputs(:, 1) = sm;

    sm_idx  = nan(size(sm, 1), 1);  % category flag
    sm_idx2 = nan(size(sm, 1), 1);  % segment id

    %% --- (1) noise removal (Akbar et al., 2018) ---
    range_thresh = 0.01;

    ind_nan        = isnan(sm);
    sm_idx(ind_nan) = 0;
    ind_valid      = find(~ind_nan);
    sm_valid       = sm(ind_valid);

    sm_var_thresh = (max(sm_valid) - min(sm_valid)) * range_thresh;
    dsm_valid     = diff(sm_valid);
    dsm_valid(abs(dsm_valid) <= sm_var_thresh) = NaN;

    noise_pos       = find(isnan(dsm_valid));
    noise_pos       = ind_valid(noise_pos + 1);   % flag the later one
    sm(noise_pos)   = NaN;
    sm_idx(noise_pos) = 1;

    %% --- (2) successive valid observations ---
    ind2  = find(~isnan(sm));
    sm2   = sm(ind2);
    tt2   = tt(ind2);

    dsm   = diff(sm2);
    dt    = diff(tt2);

    %% --- (3) inter-observation precipitation (thresholded sum) ---
    pp_inter = nan(size(dsm, 1), 1);
    for i = 1:size(dsm, 1)
        seg_pp        = sum(pp(ind2(i) : ind2(i+1) - 1), 1, 'omitnan');
        seg_pp(seg_pp <= pp_min) = 0;
        pp_inter(i)   = seg_pp;
    end

    %% --- (4) drop limbs that violate dry-down conditions ---
    dsm(pp_inter > 0)         = NaN;   % rainfall present
    dsm(dsm >= 0)             = NaN;   % soil moisture must decrease
    dsm(hours(dt) > hour_dif) = NaN;   % observation gap too large
    dsm(sm2 > phi)            = NaN;   % unphysical (above porosity)

    invalid_dsm = find(isnan(dsm));
    invalid_sm2 = unique([invalid_dsm; invalid_dsm + 1]);
    sm_idx(ind2(invalid_sm2)) = 2;

    %% --- (5) keep segments with at least `consec` observations ---
    valid_mask = ~isnan(dsm);
    transitions = diff([0; valid_mask; 0]);
    seg_starts  = find(transitions ==  1);
    seg_ends    = find(transitions == -1);
    seg_length  = seg_ends - seg_starts;        % number of limbs

    keep = (seg_length + 1) >= consec;
    short = ~keep;

    if any(short)
        short_idx = [seg_starts(short), seg_ends(short)];
        sm_idx(ind2(short_idx(:))) = 3;
    end

    seg_starts = seg_starts(keep);
    seg_ends   = seg_ends(keep);

    %% --- (6) build per-limb loss table ---
    if isempty(seg_starts)
        paras_dd = zeros(1, 3);
        lb       = [];
        return;
    end

    seg_info = [seg_starts, seg_ends, (1:numel(seg_starts)).'];
    sm2_dd   = [];

    for j = 1:size(seg_info, 1)
        rng_sm  = ind2(seg_info(j, 1) : seg_info(j, 2));
        sm_idx (rng_sm) = 4;
        sm_idx2(rng_sm) = j;

        sm_initial = sm2 (seg_info(j, 1) : seg_info(j, 2) - 1);
        dsm_seg    = dsm (seg_info(j, 1) : seg_info(j, 2) - 1);
        dt_seg_d   = hours(dt(seg_info(j, 1) : seg_info(j, 2) - 1)) / 24;
        loss_seg   = abs(dsm_seg) ./ dt_seg_d;

        seg_id = j * ones(numel(loss_seg), 1);
        sm2_dd = [sm2_dd; ...
                  seg_id, sm_initial, dsm_seg, dt_seg_d, loss_seg]; %#ok<AGROW>
    end

    lb = sortrows(sm2_dd, 2);
    p1 = min(lb(1,   2));
    p2 = max(lb(end, 2));

    paras_dd = [size(seg_info, 1), p1, p2];

    sm_outputs(:, 2) = sm_idx;
    sm_outputs(:, 3) = sm_idx2;
end