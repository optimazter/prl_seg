function out = ndiCG_auto(params)
% Nonlinear Dipole Inversion. Conjugate Gradient Descent solver.
% This is a faster alternative to ndi(), with automatic stopping
% by using an analysis of the frequency domain.
% Please cite Milovic C, et al. ESMRMB 2020:L01.73. and Milovic C, et al. ISMRM 2021:p3982.
%
% Parameters: params - structure with 
% Required fields:
% params.input: local field map, in radians
% params.K: dipole kernel in the frequency space
% Optional fields:
% params.alpha: regularization weight (use small values for stability)
% params.maxOuterIter: maximum number of iterations (use a large number, i.e 100)
% params.voxelSize: spatial resolution. 
% params.weight: data fidelity spatially variable weight (recommended = magnitude_data).
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
        x = params.weight.*params.input;
    else
        x = zeros(N, 'single');
    end

[m1, m2] = create_freqmasks_auto( voxelsize, kernel );

try
if isGPU 
    disp('GPU enabled');
    phase = gpuArray(phase);
    kernel = gpuArray(kernel);
    x = gpuArray(x);

    weight = gpuArray(weight);
    alpha = gpuArray(alpha);
    num_iter = gpuArray(num_iter);
    m1 = gpuArray(m1);
    m2 = gpuArray(m2);
end
catch
    disp('WARNING: GPU disabled');
end
    

fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', 'Iter', 'Update', 'Stepsize',...
                                   'Energy M1', 'Energy M2');

phix = susc2field(kernel,x);
dx = -susc2field( conj(kernel),weight.*sin(phix-phase ) ) - alpha*x;


B = dx.*dx;
A = dx.*(susc2field(conj(kernel),weight.*sin(susc2field(kernel,dx))) +alpha*dx);
tau = sum(B(:))/(sum(A(:))+eps);

x_prev = x;
x = x_prev + tau*dx;
x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
fprintf('%3d\t%10.4f\t%10.4f\t%10s\t%10s\n', 0, x_update, tau, '-', '-');

s = dx;
%tic
for t = 1:num_iter
    
    % update x : susceptibility estimate
    x_prev = x;
    phix = susc2field(kernel,x);
    dx_prev = dx;
    dx = -susc2field( conj(kernel),weight.*sin(phix-phase ) ) - alpha*x;
    
    betaPR = max(sum(dx(:).*(dx(:)-dx_prev(:)))/(sum(dx(:).*dx(:))+eps),0);
    
    s = dx + betaPR*s;
    
    
    B = s.*dx;
    A = s.*(susc2field(conj(kernel),weight.*sin(susc2field(kernel,s))) +alpha*s);
    tau = sum(B(:))/(sum(A(:))+eps);
    
    x = x_prev + tau*s;
    
    if isShowIters
        imagesc3d2(x, N/2, 31, [90,90,-90], [-1,1], [], ['NDICG - Iteration: ', num2str(t), '   Update: ', num2str(x_update)] )
        drawnow;
    end
    
    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    [e1(t), e2(t)] = compute_freqe_auto(x,m1,m2);

    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', t, x_update, tau, e1(t), e2(t));
    
    if e1(t) > e2(t) && t > 3
        disp('Automatic stop reached!')
        break
    end
    
    
end
%out.x = x_prev;
out.time = toc;toc
out.x = gather(x);
out.iter = gather(t);
out.e1 = gather(e1);
out.e2 = gather(e2);



end
