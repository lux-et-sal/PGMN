function [net, M, normParams, info] = mdn_load_model(model_file)
% mdn_load_model  Restore the trained PGMN network from disk.
%
%   [net, M, normParams, info] = mdn_load_model(model_file)
%
% Input
%   model_file  path to the shipped model, pretrained/best_model_M<M>.mat
%
% Outputs
%   net         dlnetwork with the trained weights restored
%   M           number of mixture components
%   normParams  normalization statistics (calibration period)
%   info        remaining fields of the file (patch_info, performance, ...)
%
% WHY NOT JUST load(...).net
%   The custom ConvLSTMLayer cannot be round-tripped through a saved
%   dlnetwork: MATLAB warns, substitutes a default layer, and you end up
%   with a network that runs but predicts nonsense. The training code
%   therefore stores the weights as plain arrays alongside the layer and
%   parameter names (save_method = 'learnables_direct'). This function
%   rebuilds the architecture with mdn_build_network and matches the
%   stored values back by (Layer, Parameter) name, so row order in the
%   tables is irrelevant.
%
%   Port of the authors' load_net_learnables.m. Any mismatch is an error
%   here rather than a warning: a partially restored network is worse than
%   no network, because it fails silently.
%
% Requires ConvLSTMLayer.m on the MATLAB path.

    assert(exist(model_file, 'file') == 2, 'Model file not found: %s', model_file);
    S = load(model_file);

    assert(isfield(S, 'save_method') && strcmp(S.save_method, 'learnables_direct'), ...
        ['%s is not in learnables_direct format. A model saved as a plain ' ...
         'dlnetwork object has an unusable ConvLSTM layer -- regenerate it ' ...
         'with trimBestModel.m from a training run.'], model_file);
    assert(isfield(S, 'learnables_vals') && isfield(S, 'input_size'), ...
        '%s is missing learnables_vals or input_size.', model_file);

    M          = double(S.M);
    normParams = S.normParams;

    net = mdn_build_network(S.input_size, M);
    net = restore_table(net, 'Learnables', S.learnables_table, S.learnables_vals);
    if isfield(S, 'state_vals') && ~isempty(S.state_vals)
        net = restore_table(net, 'State', S.state_table, S.state_vals);
    end

    info = rmfield(S, intersect(fieldnames(S), ...
        {'learnables_table','learnables_vals','state_table','state_vals','net'}));
end

%% ---------------------------------------------------------------------
function net = restore_table(net, prop, tbl, vals)
    T = net.(prop);
    missing = strings(0,1);
    for i = 1:height(T)
        j = find(strcmp(tbl.Layer, T.Layer{i}) & ...
                 strcmp(tbl.Parameter, T.Parameter{i}), 1);
        if isempty(j)
            missing(end+1) = string(T.Layer{i}) + "/" + string(T.Parameter{i}); %#ok<AGROW>
            continue;
        end
        T.Value{i} = dlarray(single(vals{j}));
    end
    assert(isempty(missing), ...
        'mdn_load_model: %d %s entries absent from the saved model (%s).', ...
        numel(missing), prop, strjoin(cellstr(missing), ', '));
    net.(prop) = T;
end
