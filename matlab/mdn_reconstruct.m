function out = mdn_reconstruct(net, M, normParams, S, seq_len, useGPU)
% mdn_reconstruct  Run PGMN over the shipped evaluation block.
%
%   out = mdn_reconstruct(net, M, normParams, S, seq_len, useGPU)
%
% Inputs
%   net         dlnetwork from mdn_load_model
%   M           number of mixture components
%   normParams  normalization statistics from the same file
%   S           struct loaded from data/sample_patch.mat
%   seq_len     ConvLSTM sequence length (10; must match training)
%   useGPU      logical, optional; defaults to canUseGPU()
%
% Output struct, all [64 64 995] unless stated
%   smap_true      SMAP observation, m^3/m^3 (NaN where unobserved)
%   smap_pred      PGMN reconstruction, m^3/m^3
%   wbm            WBM baseline, m^3/m^3
%   residual_std   predicted total standard deviation, m^3/m^3
%   mask           logical, true where SMAP was observed
%   xval_idx       [1 995] index into the 1004-step block for each output step
%   time           [995 1] physical date of each output step
%
% NOTES
%   - The model takes four channels. /XVal stores three, and the fourth,
%     ResAvail(t-1), is appended here. The order is precip, WBM,
%     residual(t-1), ResAvail. A wrong order raises no error and gives
%     wrong results.
%   - WBM and SMAP are read back from the quantized arrays. Reading them
%     from other sources shifts the metrics in the fourth decimal.
%   - A reconstruction at or below zero takes the WBM value.
%
% MIXTURE HEAD
%   The 3M output channels are [mu(1:M), sigma_raw(M+1:2M), pi_raw(2M+1:3M)].
%   The point prediction and its standard deviation follow Equations (5)
%   and (6) of the paper.

    if nargin < 5 || isempty(seq_len), seq_len = 10; end
    if nargin < 6 || isempty(useGPU)
        % Falls back to CPU without a GPU or Parallel Computing Toolbox.
        useGPU = false;
        try, useGPU = canUseGPU(); catch, end
    end

    %% ---------- dequantize -------------------------------------------
    X3 = dequantize(S.XVal,        S.scale_factor, S.fill_value);   % [H W 3 T]
    Y  = dequantize(S.YVal_Direct, S.scale_factor, S.fill_value);   % [H W T]
    mask_full = logical(S.MaskVal);

    % Fourth channel: observation availability at t-1, not quantized.
    RA = single(S.ResAvailVal);
    X  = cat(3, X3, reshape(RA, size(RA,1), size(RA,2), 1, size(RA,3)));
    clear X3 RA;

    Y(~mask_full) = NaN;                       % unobserved days carry no target

    [H, W, C, T] = size(X);
    assert(C == 4, 'mdn_reconstruct: expected 4 input channels, got %d', C);

    %% ---------- sequence bookkeeping ---------------------------------
    xval_idx      = seq_len:T;                 % scored positions in the block
    num_sequences = numel(xval_idx);
    assert(num_sequences > 0, 'Evaluation block shorter than seq_len.');

    batch_size  = 256;
    num_batches = ceil(num_sequences / batch_size);

    smap_pred_seq    = zeros(H, W, num_sequences, 'single');
    smap_true_seq    = zeros(H, W, num_sequences, 'single');
    wbm_seq          = zeros(H, W, num_sequences, 'single');
    residual_std_seq = zeros(H, W, num_sequences, 'single');
    mask_seq         = false(H, W, num_sequences);

    fprintf('  mdn_reconstruct: %d sequences, %d batches, M=%d, GPU=%s\n', ...
            num_sequences, num_batches, M, mat2str(useGPU));

    %% ---------- inference --------------------------------------------
    for bi = 1:num_batches
        b0  = (bi - 1) * batch_size + 1;
        b1  = min(bi * batch_size, num_sequences);
        bsz = b1 - b0 + 1;

        XBatch = zeros(H, W, C, seq_len, bsz, 'single');
        for b = 1:bsz
            t_end   = xval_idx(b0 + b - 1);
            t_start = t_end - seq_len + 1;
            XBatch(:, :, :, :, b) = X(:, :, :, t_start:t_end);
        end

        XBatch = dlarray(XBatch, 'SSCTB');
        if useGPU, XBatch = gpuArray(XBatch); end

        pred = gather(extractdata(predict(net, XBatch)));      % [H W 3M bsz]

        mu_all    = pred(:, :, 1:M,       :);
        sigma_raw = pred(:, :, M+1:2*M,   :);
        pi_raw    = pred(:, :, 2*M+1:3*M, :);

        sigma_all = log(1 + exp(min(sigma_raw, 10))) + 1e-6;   % softplus
        pi_exp    = exp(pi_raw - max(pi_raw, [], 3));          % softmax
        pi_all    = pi_exp ./ (sum(pi_exp, 3) + 1e-10);

        mu_pred     = sum(pi_all .* mu_all, 3);
        mu_rep      = repmat(mu_pred, [1, 1, M, 1]);
        sigma_total = sqrt(sum(pi_all .* ((mu_all - mu_rep).^2 + sigma_all.^2), 3));

        for b = 1:bsz
            k     = b0 + b - 1;
            t_end = xval_idx(k);

            residual_pred = mu_pred(:, :, 1, b) * normParams.residual_std ...
                            + normParams.residual_mean;

            wbm = X(:, :, 2, t_end) * normParams.wbm_std + normParams.wbm_mean;

            smap_corrected = wbm - residual_pred;
            % At or below zero, keep the WBM value (see header).
            smap_pred_seq(:, :, k) = (smap_corrected > 0) .* smap_corrected ...
                                   + (smap_corrected <= 0) .* wbm;

            smap_true_seq(:, :, k)    = Y(:, :, t_end) * normParams.smap_std ...
                                        + normParams.smap_mean;
            wbm_seq(:, :, k)          = wbm;
            residual_std_seq(:, :, k) = sigma_total(:, :, 1, b) * normParams.residual_std;
            mask_seq(:, :, k)         = mask_full(:, :, t_end);
        end
    end

    %% ---------- pack --------------------------------------------------
    out = struct( ...
        'smap_true',    smap_true_seq, ...
        'smap_pred',    smap_pred_seq, ...
        'wbm',          wbm_seq, ...
        'residual_std', residual_std_seq, ...
        'mask',         mask_seq, ...
        'xval_idx',     xval_idx);

    if isfield(S, 'time'), out.time = S.time(xval_idx); end
end

%% ---------------------------------------------------------------------
function d = dequantize(raw, scale_factor, fill_value)
% Fill values become 0 on the normalized scale, as in training.
    d = double(raw);
    invalid = (d == double(fill_value));
    d = d / double(scale_factor);
    d(invalid) = 0;
    d = single(d);
end
