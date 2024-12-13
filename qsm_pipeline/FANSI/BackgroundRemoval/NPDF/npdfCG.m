function out = npdfCG(params)
% Nonlinear Projection onto Dipole Fields (PDF). Conjugate Gradient Descent solver.
%
% Parameters: params - structure with 
% Required fields:
% params.input: total field map, in radians
% params.K: dipole kernel in the frequency space
% params.mask: ROI binary mask
% Optional fields:
% params.alpha: regularization weight (use small values for stability)
% params.maxOuterIter: maximum number of iterations (recommended = ?)
% params.weight: data fidelity spatially variable weight (recommended = magnitude_data).
% params.precond: preconditionate solution (for stability)
% params.isShowIters: show intermediate results for each iteration.
% params.isGPU: activate GPU acceleration (default = true).
%
% Output: out - structure with the following fields:
% out.local: local field map
% out.x: calculated susceptibility map, in radians
% out.time: total elapsed time (including pre-calculations)
%
% Last modified by Carlos Milovic in 2021.10.12


tic

    % Required parameters
    kernel = params.K;
    mask = single(params.mask);
    phase = params.input;

    if isfield(params,'alpha')
         alpha = params.alpha;
    else
        alpha = eps;
    end
    
    N = size(params.input);

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
        isPrecond = true;
    end
    
    if ~isfield(params,'isGPU')
        isGPU = true;
    else
        isGPU = params.isGPU;
    end
    
    
    if isPrecond        
        if isfield(params,'x0')
            x = params.x0;
        else
            %x = params.weight.*params.input;
            %x = 9.395*(1-mask);
            se = strel('sphere',2);
            mask2=imdilate(mask,se);
            [ phi_e ] = backmodel( 1-mask2, 0.0, 0, [0 0 0]);
            [ background, kappa ] = fitmodels( params.input, mask, params.weight, phi_e );
            x = kappa(5)*(1-mask2);
            display(['Background scaling: ', num2str( kappa(5) ) ]);
        end
    else
        x = zeros(N, 'single');
    end
try
if isGPU 
    display('GPU enabled');
    phase = gpuArray(phase);
    kernel = gpuArray(kernel);
    x = gpuArray(x);
    mask = gpuArray(mask);

    weight = gpuArray(weight);
    alpha = gpuArray(alpha);
    num_iter = gpuArray(num_iter);

end
catch
    disp('WARNING: GPU disabled');
end
    
    
% Initiate with a gradient descent, with automatic step by line search.
    phix = susc2field(kernel,x);
    %dx = -(1-mask).*susc2field( conj(kernel),weight.*sin(phix-phase ) ) - alpha*(1-mask).*x;
    dx = -(1-mask).*susc2field( conj(kernel),weight.*(phix-phase ) ) - alpha*(1-mask).*x;


    B = dx.*dx;
    %A = dx.*(susc2field(conj(kernel),weight.*sin(susc2field(kernel,dx))) +alpha*dx);
    A = dx.*(susc2field(conj(kernel),weight.*(susc2field(kernel,dx))) +alpha*dx);
    tau = sum(B(:))/(sum(A(:))+eps);

    x_prev = (1-mask).*x;
    x = x_prev + tau*(1-mask).*dx;
    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    disp(['Iter: ', num2str(0), '   Update: ', num2str(x_update)])

    s = dx;
%tic
% Start CG steps
for t = 1:num_iter
%     if mod(t,50)==0
%     x = gpuArray(medfilt3(gather(x)));
%     end
    %alpha=alpha*2^(-0.1);
    % update x : susceptibility estimate
    x_prev = (1-mask).*x;
    phix = susc2field(kernel,x);
    dx_prev = dx;
    if t < 10001
        dx = -(1-mask).*susc2field( conj(kernel),weight.*(phix-phase ) ) - alpha*(1-mask).*x;
    else
        dx = -(1-mask).*susc2field( conj(kernel),weight.*sin(phix-phase ) ) - alpha*(1-mask).*x;
    end
    
    betaPR = max(sum(dx(:).*(dx(:)-dx_prev(:)))/(sum(dx(:).*dx(:))+eps),0); % Automatic reset
    
    s = dx + betaPR*s;
    
    
    B = s.*dx;
    if t < 10001
        A = s.*(susc2field(conj(kernel),weight.*(susc2field(kernel,s))) +alpha*s);
    else
        A = s.*(susc2field(conj(kernel),weight.*sin(susc2field(kernel,s))) +alpha*s);
    end
    tau = sum(B(:))/(sum(A(:))+eps);
    
    x = x_prev + tau*(1-mask).*s;
    
    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    disp(['Iter: ', num2str(t), '   Update: ', num2str(x_update)])
    
    if isShowIters
        %out.err(t) = compute_rmse((phase-phix).*mask,params.gt.*mask);
        imagesc3d2(phase-phix, N/2, 42, [90,90,90], [-0.3,0.3], [], ['PDF-CG - Iteration: ', num2str(t),'   Update: ', num2str(x_update)] )% '   Err: ', num2str(out.err(t))] )%
        drawnow;
    end
end


out.time = toc;toc
out.iter = t;


if isGPU
    out.x = gather(x);
    out.phi = gather(real(ifftn(kernel.*fftn((1-mask).*x))));
else
    out.x = x;
    out.phi = real(ifftn(kernel.*fftn((1-mask).*x)));
end

out.local = params.input;%(params.input - out.phi);
for i = 1:25
    out_old = out.local;
    out.local = out.local + 2*pi*round( (out.phi - out.local)/(2*pi) );

    if sum(abs(out_old(:)-out.local(:))) < 1
        break;
    end
end
out.local = out.local-out.phi;

end
