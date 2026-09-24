function setSeed(seed)
% setSeed  Set both CPU and GPU random seeds for reproducible runs.
%
%   setSeed(seed)
%
% MDN inference with a frozen network is deterministic on the same
% hardware, so seeding mainly affects (a) any future re-training and
% (b) tie-breaking in non-deterministic GPU kernels. Calling this before
% inference is cheap insurance.

    if nargin < 1, seed = 42; end
    rng(seed, 'twister');
    % gpurng needs Parallel Computing Toolbox and a GPU; without them there
    % is no GPU stream to seed and nothing here needs to happen.
    try
        if canUseGPU(), gpurng(seed, 'Threefry'); end
    catch
    end
end
