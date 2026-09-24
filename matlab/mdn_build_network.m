function net = mdn_build_network(inputSize, M, verbose)
% mdn_build_network  Rebuild the PGMN architecture (untrained).
%
%   net = mdn_build_network(inputSize, M)
%   net = mdn_build_network(inputSize, M, true)     % print a summary
%
% Inputs
%   inputSize  [H W C], e.g. [64 64 4]
%   M          number of mixture components
%   verbose    optional, default false
%
% Output
%   net        dlnetwork with randomly initialized weights
%
% WHY THIS FILE EXISTS
%   The custom ConvLSTMLayer does not survive being saved inside a
%   dlnetwork object -- reloading it silently substitutes a default layer
%   and the weights are lost. The training code works around this by
%   storing the Learnables and State tables as plain values
%   (save_method = 'learnables_direct'). Restoring a model therefore means
%   rebuilding the architecture here and pouring the stored values back in;
%   mdn_load_model.m does the second half.
%
%   This is a verbatim port of the authors' training-time network builder.
%   Layer names matter: they are the keys used to match the stored weights.
%   Do not rename a layer without regenerating the shipped model file.
%
% ARCHITECTURE
%   input   [H W C] sequence, no normalization (inputs arrive normalized)
%   encoder 3 x (conv3x3 stride 2 -> batchnorm -> relu -> dropout 0.3)
%           filters 16 -> 24 -> 32, so [64 64] -> [32 32] -> [16 16] -> [8 8]
%   ConvLSTM 32 filters, 3x3, custom layer
%   decoder 3 x (transposed conv 4x4 stride 2 -> batchnorm -> relu[ -> dropout])
%           filters 32 -> 24 -> 16, back to [64 64]
%   head    conv 1x1 with 3M channels = [mu(1:M), sigma_raw(M+1:2M), pi_raw(2M+1:3M)]
%
%   Dropout is inactive at inference (predict), so the forward pass is
%   deterministic given the weights.

    if nargin < 3 || isempty(verbose), verbose = false; end

    H = inputSize(1);  W = inputSize(2);  C = inputSize(3);

    enc1_filters     = 16;
    enc2_filters     = 24;
    enc3_filters     = 32;
    convlstm_filters = 32;
    dropout_rate     = 0.3;

    encoderLayers = [
        sequenceInputLayer([H, W, C], 'Name', 'input', 'Normalization', 'none')

        convolution2dLayer(3, enc1_filters, 'Padding', 'same', 'Stride', 2, 'Name', 'enc_conv1')
        batchNormalizationLayer('Name', 'enc_bn1')
        reluLayer('Name', 'enc_relu1')
        dropoutLayer(dropout_rate, 'Name', 'enc_drop1')

        convolution2dLayer(3, enc2_filters, 'Padding', 'same', 'Stride', 2, 'Name', 'enc_conv2')
        batchNormalizationLayer('Name', 'enc_bn2')
        reluLayer('Name', 'enc_relu2')
        dropoutLayer(dropout_rate, 'Name', 'enc_drop2')

        convolution2dLayer(3, enc3_filters, 'Padding', 'same', 'Stride', 2, 'Name', 'enc_conv3')
        batchNormalizationLayer('Name', 'enc_bn3')
        reluLayer('Name', 'enc_relu3')
        dropoutLayer(dropout_rate, 'Name', 'enc_drop3')
    ];

    convlstmLayer = ConvLSTMLayer(convlstm_filters, 3, 'convlstm');

    decoderLayers = [
        transposedConv2dLayer(4, enc3_filters, 'Cropping', 'same', 'Stride', 2, 'Name', 'dec_tconv1')
        batchNormalizationLayer('Name', 'dec_bn1')
        reluLayer('Name', 'dec_relu1')
        dropoutLayer(dropout_rate, 'Name', 'dec_drop1')

        transposedConv2dLayer(4, enc2_filters, 'Cropping', 'same', 'Stride', 2, 'Name', 'dec_tconv2')
        batchNormalizationLayer('Name', 'dec_bn2')
        reluLayer('Name', 'dec_relu2')
        dropoutLayer(dropout_rate, 'Name', 'dec_drop2')

        transposedConv2dLayer(4, enc1_filters, 'Cropping', 'same', 'Stride', 2, 'Name', 'dec_tconv3')
        batchNormalizationLayer('Name', 'dec_bn3')
        reluLayer('Name', 'dec_relu3')

        convolution2dLayer(1, M*3, 'Padding', 'same', 'Name', 'output_conv')
    ];

    lgraph = layerGraph();
    lgraph = addLayers(lgraph, encoderLayers);
    lgraph = addLayers(lgraph, convlstmLayer);
    lgraph = addLayers(lgraph, decoderLayers);
    lgraph = connectLayers(lgraph, 'enc_drop3', 'convlstm');
    lgraph = connectLayers(lgraph, 'convlstm',  'dec_tconv1');

    net = dlnetwork(lgraph);

    if verbose
        n = 0;
        for i = 1:height(net.Learnables)
            n = n + numel(net.Learnables.Value{i});
        end
        fprintf('  mdn_build_network: [%d %d %d] -> M=%d, %d learnables\n', ...
                H, W, C, M, n);
    end
end
