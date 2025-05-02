function out = ldi(params)
% Linear Dipole Inversion. Gradient Descent solver.
% This is a Tikhonov regularized iterative linear QSM solver.
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


    if isfield(params,'alpha')
         alpha = params.alpha;
    else
        alpha = 1E-6;
    end
    
    if isfield(params,'tau')
         tau = params.tau;
    else
        tau = 1.0;
    end
    
    if isfield(params,'N')
         N = params.N;
    else
        N = size(params.input);
    end

    if isfield(params,'maxOuterIter')
        num_iter = params.maxOuterIter;
    else
        num_iter = 500;
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
    
    if isfield(params,'GT')
        isGT = true;
        GT = params.GT;
        if isfield(params,'mask')
            mask = params.mask;
        else
            mask = single(weight > 0);
        end
        if isfield(params,'scale')
            phs_scale = params.scale;
        else
            phs_scale = 1.0;
        end
        
    else
        isGT = false;
    end
    
    
    if isPrecond   
        x =params.weight.*params.input;
    else
        x = zeros(N, 'single');
    end

    kernel = params.K;
    phase = params.input;

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
    
    if isGT
        GT = gpuArray(GT);
        mask = gpuArray(mask);
        phs_scale = gpuArray(phs_scale);
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
    x = x_prev - tau*susc2field( conj(kernel),weight.*(phix-phase ) ) - tau*alpha*x;
    
    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    fprintf('%3d\t%10.4f\n', t, x_update);
    
    if isShowIters
        imagesc3d2(x, N/2, 31, [90,90,-90], [-0.1,0.1], [], ['LDI - Iteration: ', num2str(t), '   Update: ', num2str(x_update)] )
        drawnow;
    end

    if isGT
        rmse = compute_rmse(x.*mask/phs_scale,GT.*mask);
        out.rmse(t) = gather(rmse);
        xsim = compute_xsim(x.*mask/phs_scale,GT.*mask);
        out.xsim(t) = gather(xsim);
    end
    
    out.update(t) = gather(x_update);
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
