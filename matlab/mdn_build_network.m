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
% NOTES
%   mdn_load_model.m calls this to rebuild the network before restoring
%   the stored weights. Layer names are the keys for that match, so do not
%   rename a layer.
%
%   Architecture and training settings are given in Table S1 of the
%   paper. Dropout is inactive at inference, so the forward pass is
%   deterministic.

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
