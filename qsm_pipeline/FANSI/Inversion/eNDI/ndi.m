function out = ndi(params)
% Nonlinear Dipole Inversion. Gradient Descent solver.
% Based on Polak D, et al. NMR Biomed 2020.
%
% Parameters: params - structure with 
% Required fields:
% params.input: local field map, in radians
% params.K: dipole kernel in the frequency space
% Optional fields:
% params.alpha: regularization weight (use small values for stability)
% params.maxOuterIter: maximum number of iterations
% params.weight: data fidelity spatially variable weight (recommended = magnitude_data).
% params.tau: gradient descent rate
% params.precond: preconditionate solution (for stability)
% params.isShowIters: show intermediate results for each iteration.
% params.isGPU: activate GPU acceleration (default = true).
%
% Output: out - structure with the following fields:
% out.x: calculated susceptibility map, in radians
% out.time: total elapsed time (including pre-calculations)
%
% Last modified by Carlos Milovic in 2021.10.12


tic

    % Required parameters
    kernel = params.K;
    phase = params.input;

    if isfield(params,'alpha')
         alpha = params.alpha;
    else
        alpha = 1E-5;
    end
    
    if isfield(params,'tau')
         tau = params.tau;
    else
        tau = 2.0; % Accelerate it slightly. Too large values may cause fast divergence.
    end
    
    N = size(params.input);

    if isfield(params,'maxOuterIter')
        num_iter = params.maxOuterIter;
    else
        num_iter = 100;
    end
    
    if isfield(params,'weight')
        weight = params.weight;
    else
        weight = ones(N);
    end
    weight = weight.*weight;
    

    if isfield(params,'isShowIters')
        isShowIters = params.isShowIters;
    else
        isShowIters = false;
    end
    
    if isfield(params,'mask')
            mask = params.mask;
    else
            mask = single(weight > 0);
    end
        
    if isfield(params,'GT')
        isGT = true;
        GT = params.GT;
        
        if isfield(params,'scale')
            phs_scale = params.scale;
        else
            phs_scale = 1.0;
        end
        if isfield(params,'maskGT')
            maskGT = params.maskGT;
        else
            maskGT = mask;
        end
        
    else
        isGT = false;
    end

    if isfield(params,'isPrecond')
        isPrecond = params.isPrecond;
    else
        isPrecond = false;
    end
    
    if ~isfield(params,'isGPU')
        isGPU = true;
    else
        isGPU = params.isGPU;
    end
    
    
    if isPrecond  
        if isfield(params,'precond')
            x = params.precond;
        else
            x = params.weight.*params.input;
        end 
    else
        x = zeros(N, 'single');
    end

try
if isGPU 
    disp('GPU enabled');
    phase = gpuArray(phase);
    kernel = gpuArray(kernel);
    x = gpuArray(x);

    weight = gpuArray(weight);
    alpha = gpuArray(alpha);
    tau = gpuArray(tau);
    num_iter = gpuArray(num_iter);
    
    mask = gpuArray(mask);
    if isGT
        GT = gpuArray(GT);
        phs_scale = gpuArray(phs_scale);
        maskGT = gpuArray(maskGT);
    end
end
catch
    disp('WARNING: GPU disabled');
end
    

fprintf('%3s\t%10s\n', 'Iter', 'Update');

%tic
for t = 1:num_iter
    % update x : susceptibility estimate
    x_prev = x;
    phix = susc2field(kernel,x);
    x = x_prev - tau*susc2field( conj(kernel),weight.*sin(phix-phase ) ) - tau*alpha*x;
    
    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    fprintf('%3d\t%10.4f\n', t, x_update);
    
    if isShowIters
        imagesc3d2(x, N/2, 131, [90,90,-90], [-0.1,0.1], [], ['NDI - Iteration: ', num2str(t), '   Update: ', num2str(x_update)] )
        drawnow;
    end

    if isGT
        rmse = compute_rmse(x.*maskGT/phs_scale,GT.*maskGT);
        out.rmse(t) = gather(rmse);
    end
end
out.time = toc;toc

if isGPU
    out.x = gather(x);
    out.iter = gather(t);
else
    out.x = x;
    out.iter = t;
end

end
