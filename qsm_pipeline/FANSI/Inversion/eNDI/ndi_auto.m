function out = ndi_auto(params)
% Nonlinear Dipole Inversion. Gradient Descent solver with an automatic stopping criterion.
% Based on Polak D, et al. NMR Biomed 2020. and Milovic C, et al. ISMRM 2021:p3982.
%
% Parameters: params - structure with 
% Required fields:
% params.input: local field map, in radians
% params.K: dipole kernel in the frequency space
% Optional fields:
% params.alpha: regularization weight (use small values for stability)
% params.maxOuterIter: maximum number of iterations (use a large number, i.e 1000)
% params.voxelSize: spatial resolution. 
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
        alpha = 1E-6;
    end
    
    if isfield(params,'voxelSize')
         voxelsize = params.voxelSize;
    else
        voxelsize = [1 1 1];
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
    
    
    if isPrecond   
        x = 3*params.weight.*phase;
    else
        x = zeros(N, 'single');
    end


[m1, m2] = create_freqmasks_auto( voxelsize, kernel );

try
if isGPU 
    disp('GPU enabled');
    kernel = gpuArray(single(kernel));
    x = gpuArray(x);

    weight = gpuArray(single(weight));
    alpha = gpuArray(alpha);
    tau = gpuArray(tau);
    num_iter = gpuArray(num_iter);
    m1 = gpuArray(single(m1));
    m2 = gpuArray(single(m2));
end
catch
    disp('WARNING: GPU disabled');
end
    

fprintf('%3s\t%10s\t%10s\t%10s\n', 'Iter', 'Update', 'Energy M1', 'Energy M2');
%tic
for t = 1:num_iter
    % update x : susceptibility estimate
    x_prev = x;
    x = x_prev - tau*susc2field( conj(kernel),weight.*sin(susc2field(kernel,x)-phase ) ) - tau*alpha*x;
    
    %imagesc3d2(x.*mask/phs_scale, N/2, 31, [90,90,-90], [-0.10,0.10], [], 'ndi')
    
    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    [e1(t), e2(t)] = compute_freqe_auto(x,m1,m2);

    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\n', t, x_update, e1(t), e2(t));
    
    if isShowIters
        imagesc3d2(x, N/2, 31, [90,90,-90], [-1,1], [], ['NDICG - Iteration: ', num2str(t), '   Update: ', num2str(x_update)] )
        drawnow;
    end
    
    if e1(t) > e2(t) && t > 3
        disp('Automatic stop reached!')
        break
    end
    out.update(t) = gather(x_update);
end
out.time = toc;toc
out.x = gather(x);
out.iter = gather(t);
out.e1 = gather(e1);
out.e2 = gather(e2);



end
