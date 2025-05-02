function out = wL1TV(params)
% Linear L1-norm QSM and Total Variation regularization 
% with spatially variable fidelity and regularization weights.
% The L1-norm data fidelity term (PI-QSM) is more robust against phase inconsistencies than
% standard L2-norm (FANSI). 
% This uses ADMM to solve the functional.
%
% Parameters: params - structure with 
% Required fields:
% params.input: local field map
% params.K: dipole kernel in the frequency space
% params.alpha1: gradient penalty (L1-norm) or regularization weight
% Optional fields:
% params.mu1: gradient consistency weight (ADMM weight, recommended = 100*alpha1)
% params.mu2: fidelity consistency weight (ADMM weight, recommended value = 1.0)
% params.maxOuterIter: maximum number of iterations (recommended = 150)
% params.tolUpdate: convergence limit, update ratio of the solution (recommended = 0.1)
% params.weight: data fidelity spatially variable weight (recommended = lambda*magnitude_data
%                or lambda*mask), with the magnitude in the [0,1] range.
%                IMPORTANT: This parameter affects the L1 proximal operation performed
%                           between the predicted magnetization and the acquired data. Unlike
%                           FANSI, this weight should be rescaled by a lambda factor to increase 
%                           or decrease the phase rejection strength. Lambda < 1 rejects more
%                           voxels with measured inconsistencies.
% params.regweight: regularization spatially variable weight.
% params.isPrecond: preconditionate solution by smart initialization (default = true)
% params.isGPU: GPU acceleration (default = true)
%
% Output: out - structure with the following fields:
% out.x: calculated susceptibility map
% out.iter: number of iterations needed
% out.time: total elapsed time (including pre-calculations)
%
% Modified by Carlos Milovic in 2019.11.27
% Modified by Carlos Milovic in 2020.07.14
% Last modified by Carlos Milovic and Patrich Fuchs in 2021.10.14

tic

    % Required parameters
    alpha = params.alpha1;
    kernel = params.K;
    Wy = params.input;

    % Optional parameters
    if isfield(params,'mu1')
         mu = params.mu1;
    else
        mu = 10*alpha;
    end
    
    if isfield(params,'mu2')
         mu2 = params.mu2;
    else
        mu2 = 1.0;
    end
    
    N = size(params.input);

    if isfield(params,'maxOuterIter')
        num_iter = params.maxOuterIter;
    else
        num_iter = 50;
    end
    
    if isfield(params,'tolUpdate')
       tolUpdate  = params.tolUpdate;
    else
       tolUpdate = 0.1;
    end

    if isfield(params,'weight')
        W = params.weight;
    else
        W = ones(N);
    end
    %weight = weight.*weight;
    
    if isfield(params,'regweight')
        regweight = params.regweight;
        if length(size(regweight)) == 3
            regweight = repmat(regweight,[1,1,1,3]);
        end
    else
        regweight = ones([N 3]);
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
    
    
    
    % Variable initialization
z_dx = zeros(N, 'single');
z_dy = zeros(N, 'single');
z_dz = zeros(N, 'single');

s_dx = zeros(N, 'single');
s_dy = zeros(N, 'single');
s_dz = zeros(N, 'single');

x = zeros(N, 'single');


    
s2 = zeros(N,'single');

alpha_over_mu = alpha/mu;



    if isPrecond
        z2 = Wy.*W;
    else
        z2 = zeros(N,'single');
    end
    
% Define the operators
[k1, k2, k3] = ndgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);

E1 = 1 - exp(2i .* pi .* k1 / N(1));
E2 = 1 - exp(2i .* pi .* k2 / N(2));
E3 = 1 - exp(2i .* pi .* k3 / N(3));



% Move variables to GPU
try
if isGPU 
    disp('GPU enabled');
    Wy = gpuArray(Wy);
    z_dx = gpuArray(z_dx);
    z_dy = gpuArray(z_dy);
    z_dz = gpuArray(z_dz);

    s_dx = gpuArray(s_dx);
    s_dy = gpuArray(s_dy);
    s_dz = gpuArray(s_dz);

    x = gpuArray(x);
    kernel = gpuArray(kernel);

    z2 = gpuArray(z2);
    s2 = gpuArray(s2);

    tolUpdate = gpuArray(tolUpdate);

    E1 = gpuArray(E1);
    E2 = gpuArray(E2);
    E3 = gpuArray(E3);

    alpha_over_mu = gpuArray(alpha_over_mu);
    regweight = gpuArray(regweight);
    mu = gpuArray(mu);
    W = gpuArray(W);
    mu2 = gpuArray(mu2);
    
end
catch
    disp('WARNING: GPU disabled');
end
    




E1t = conj(E1);
E2t = conj(E2);
E3t = conj(E3);

EE2 = E1t .* E1 + E2t .* E2 + E3t .* E3;

fprintf('%3s\t%10s\n', 'Iter', 'Update');
%tic
for t = 1:num_iter
    % update x : susceptibility estimate
    tx = E1t .* fftn(z_dx - s_dx);
    ty = E2t .* fftn(z_dy - s_dy);
    tz = E3t .* fftn(z_dz - s_dz);
    
    x_prev = x;
    x = real(ifftn( (mu * (tx + ty + tz) + mu2*conj(kernel) .* fftn(z2-s2+Wy)) ./ (eps + mu2*abs(kernel).^2 + mu * EE2) ));
    
    clear tx ty tz

    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    fprintf('%3d\t%10.4f\n', t, x_update);
    %imagesc3d2(x, [81    93    47], 42, [90,90,90], [-0.15,0.15], 0, 'wL1');
    
    if x_update < tolUpdate
        break
    end
    
    if t < num_iter
        % update z : gradient varible
        Fx = fftn(x);
        x_dx = real(ifftn(E1 .* Fx));
        x_dy = real(ifftn(E2 .* Fx));
        x_dz = real(ifftn(E3 .* Fx));
        
        z_dx = max(abs(x_dx + s_dx) - regweight(:,:,:,1)*alpha_over_mu, 0) .* sign(x_dx + s_dx);
        z_dy = max(abs(x_dy + s_dy) - regweight(:,:,:,2)*alpha_over_mu, 0) .* sign(x_dy + s_dy);
        z_dz = max(abs(x_dz + s_dz) - regweight(:,:,:,3)*alpha_over_mu, 0) .* sign(x_dz + s_dz);
    
        % update s : Lagrange multiplier
        s_dx = s_dx + x_dx - z_dx;
        s_dy = s_dy + x_dy - z_dy;            
        s_dz = s_dz + x_dz - z_dz;  
        
        clear x_dx x_dy x_dz
        % update z2 and s2 : data consistency
        z2_inner = real(ifftn(kernel.*Fx))+s2 -Wy;
        z2 = max( abs(z2_inner)-W/mu2, 0.0 ) .* sign(z2_inner) ; % proximal operation
        
        s2 = z2_inner - z2;
        clear Fx z2_inner
    end
end
% Extract output values
out.time = toc;toc

out.x = gather(x);
out.iter = gather(t);


end
