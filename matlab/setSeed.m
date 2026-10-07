function setSeed(seed)
% setSeed  Set both CPU and GPU random seeds for reproducible runs.
%
%   setSeed(seed)
%
% Inference with a trained network is deterministic on the same hardware.
% Seeding guards against nondeterministic GPU kernels.

    if nargin < 1, seed = 42; end
    rng(seed, 'twister');
    % gpurng needs Parallel Computing Toolbox and a GPU; without them there
    % is no GPU stream to seed and nothing here needs to happen.
    try
        if canUseGPU(), gpurng(seed, 'Threefry'); end
    catch
    end
end
