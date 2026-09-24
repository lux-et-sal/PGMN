classdef ConvLSTMLayer < nnet.layer.Layer & nnet.layer.Formattable
    % ConvLSTMLayer  Custom ConvLSTM layer for the PGMN architecture.
    %
    % Accepts a 5-D input of format SSCTB (spatial, spatial, channel,
    % time, batch) or SSCBT; the time and batch dimensions are reordered
    % internally so that time is always the last one. The output is the
    % hidden state after the last time step, format SSCB.
    %
    % This file must be on the MATLAB path before mdn_build_network is
    % called, since the architecture instantiates this layer by name.
    %
    % The layer is also why the trained model is shipped as stored weights
    % rather than as a saved dlnetwork: a custom layer does not survive the
    % round trip, and MATLAB substitutes a default in its place without
    % failing. mdn_load_model.m rebuilds and refills instead.

    properties
        NumFilters
        FilterSize
    end

    properties (Learnable)
        InputWeights
        RecurrentWeights
        Bias
    end

    methods
        function layer = ConvLSTMLayer(numFilters, filterSize, name)
            layer.Name = name;
            layer.Description = "ConvLSTM with " + numFilters + " filters";
            layer.Type = "ConvLSTM";
            layer.NumFilters = numFilters;
            layer.FilterSize = filterSize;
        end

        function layer = initialize(layer, layout)
            % Random initialization. Only the shapes matter for PGMN: the
            % trained values are poured in afterwards by mdn_load_model.
            idx = find(layout.Format == 'C', 1);
            if isempty(idx)
                error('ConvLSTM:NoChannelDim', 'Input layout has no channel (C) dimension.');
            end
            inputChannels = layout.Size(idx);

            % Input weights: one 3x3 kernel per (input channel, gate unit)
            sz = [layer.FilterSize, layer.FilterSize, inputChannels, 4 * layer.NumFilters];
            numIn  = inputChannels * layer.FilterSize^2;
            numOut = layer.NumFilters * layer.FilterSize^2;
            layer.InputWeights = dlarray(glorot(sz, numOut, numIn));

            % Recurrent weights
            sz = [layer.FilterSize, layer.FilterSize, layer.NumFilters, 4 * layer.NumFilters];
            numIn = layer.NumFilters * layer.FilterSize^2;
            layer.RecurrentWeights = dlarray(glorot(sz, numOut, numIn));

            % Bias, with the forget gate started at 1
            layer.Bias = dlarray(zeros(1, 1, 4 * layer.NumFilters, 'single'));
            layer.Bias(1, 1, (layer.NumFilters+1):(2*layer.NumFilters)) = 1;
        end

        function Z = predict(layer, X)
            % Forward pass. X is a 5-D dlarray, SSCTB or SSCBT; Z is the
            % hidden state after the last time step, SSCB.

            x_dims = dims(X);
            if sum(x_dims == 'S') ~= 2 || ~any(x_dims == 'C') || ~any(x_dims == 'T')
                error('ConvLSTM:InvalidInput', ...
                    'Input needs two S (spatial), one C (channel) and one T (time) dimension; got %s.', x_dims);
            end

            % Put time last (SSCBT) so each step is X(:,:,:,:,t).
            b_idx = find(x_dims == 'B', 1);
            t_idx = find(x_dims == 'T', 1);
            if ~isempty(b_idx) && b_idx > t_idx
                X = dlarray(permute(extractdata(X), [1, 2, 3, 5, 4]), 'SSCBT');
                x_dims = dims(X);
            end

            sz = size(X);
            s_idx = find(x_dims == 'S');
            H = sz(s_idx(1));
            W = sz(s_idx(2));
            T = sz(find(x_dims == 'T', 1));
            b_idx = find(x_dims == 'B', 1);
            if isempty(b_idx), B = 1; else, B = sz(b_idx); end

            % Initial states, on the same device as the input
            h = dlarray(zeros(H, W, layer.NumFilters, B, 'single'), 'SSCB');
            c = dlarray(zeros(H, W, layer.NumFilters, B, 'single'), 'SSCB');
            if isa(extractdata(X), 'gpuArray')
                h = gpuArray(h);
                c = gpuArray(c);
            end

            % Recurrence over time
            nf = layer.NumFilters;
            for t = 1:T
                xt = X(:, :, :, :, t);                                   % SSCB

                gates = dlconv(xt, layer.InputWeights, layer.Bias, 'Padding', 'same') ...
                      + dlconv(h,  layer.RecurrentWeights, 0,      'Padding', 'same');

                i_gate = sigmoid(gates(:, :, 1:nf,        :));
                f_gate = sigmoid(gates(:, :, nf+1:2*nf,   :));
                g_gate = tanh   (gates(:, :, 2*nf+1:3*nf, :));
                o_gate = sigmoid(gates(:, :, 3*nf+1:4*nf, :));

                c = f_gate .* c + i_gate .* g_gate;
                h = o_gate .* tanh(c);
            end

            Z = h;
        end
    end
end

%% ---------------------------------------------------------------------
function weights = glorot(sz, numOut, numIn)
% Glorot (Xavier) initialization
    var = 2 / (numIn + numOut);
    weights = sqrt(var) * randn(sz, 'single');
end
