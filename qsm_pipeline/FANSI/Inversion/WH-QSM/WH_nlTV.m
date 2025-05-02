function out = WH_nlTV(params)
% Nonlinear Weak Harmonics - QSM and Total Variation regularization 
% with spatially variable fidelity and regularization weights.
% This uses ADMM to solve the functional.
% This function is used to remove background field remnants from *local* field maps and
% calculate the susceptibility of tissues simultaneously.
%
% Parameters: params - structure with 
% Required fields:
% params.input: local field map
% params.K: dipole kernel in the frequency space
% params.alpha1: gradient penalty (L1-norm) or regularization weight
% Optional fields:
% params.beta: harmonic constrain weight (default value = 150)
% params.muh: harmonic consistency weight (recommended value = beta/50)
% params.mask: ROI to calculate susceptibility values (if not provided, will be calculated from 'weight')
% params.mu1: gradient consistency weight (ADMM weight, recommended = 100*alpha1)
% params.mu2: fidelity consistency weight (ADMM weight, recommended value = 1.0)
% params.maxOuterIter: maximum number of iterations (recommended for testing = 150, for correct 
%                      convergence of the harmonic field hundreds of iterations are needed)
% params.tolUpdate: convergence limit, update ratio of the solution (recommended = 0.1)
% params.weight: data fidelity spatially variable weight (recommended = magnitude_data). 
% params.regweight: regularization spatially variable weight.
% params.isPrecond: preconditionate solution by smart initialization (default = true)
% params.isGPU: GPU acceleration (default = true)
%
% Output: out - structure with the following fields:
% out.x: calculated susceptibility map
% out.phi: harmonic phase in [same range as input]
% out.iter: number of iterations needed
% out.time: total elapsed time (including pre-calculations)
%
% Modified by Carlos Milovic in 2017.06.09
% Modified by Carlos Milovic in 2020.07.11
% Last modified by Carlos Milovic and Patrich Fuchs in 2021.10.14
%

tic

    % Required parameters
alpha = params.alpha1;
kernel = params.K;
phase = params.input;

N = size(params.input);
    
    % Optional parameters
    if isfield(params,'mu1')
         mu = params.mu1;
    else
        mu = 100*alpha;
    end
    if isfield(params,'mu2')
         mu2 = params.mu2;
    else
        mu2 = 1.0;
    end
    
    if isfield(params,'muh')
         muh = params.muh;
    else
        muh = 5.0;
    end
    if isfield(params,'beta')
         beta = params.beta;
    else
        beta = 150.0;
    end
    
    if isfield(params,'weight')
        W = params.weight;
    else
        W = ones(N);
    end
    W = W.*W;
    
    if isfield(params,'mask')
         mask = params.mask;
    else
        mask = W > 0;
    end
    

    if isfield(params,'maxOuterIter')
        num_iter = params.maxOuterIter;
    else
        num_iter = 150;
    end
    
    if isfield(params,'tolUpdate')
       tolUpdate  = params.tolUpdate;
    else
       tolUpdate = 0.1;
    end

    
    if isfield(params,'regweight')
        regweight = params.regweight;
        if length(size(regweight)) == 3
            regweight = repmat(regweight,[1,1,1,3]);
        end
    else
        regweight = ones([N 3]);
    end
    
    if isfield(params,'tolDelta')
        tolDelta = params.tolDelta;
    else
        tolDelta = 1e-6;
        
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

phi_h = zeros(N, 'single');

z_h = zeros(N, 'single');
s_h = zeros(N, 'single');

    if isPrecond
        z2 = phase.*W;
    else
        z2 = zeros(N,'single');
    end
s2 = zeros(N,'single');

alpha_over_mu = alpha/mu;

% Define the operators
[k1, k2, k3] = ndgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);

E1 = 1 - exp(2i .* pi .* k1 / N(1));
E2 = 1 - exp(2i .* pi .* k2 / N(2));
E3 = 1 - exp(2i .* pi .* k3 / N(3));



% Move variables to GPU
try
if isGPU 
    disp('GPU enabled!');
    phase = gpuArray(phase);
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
    
    phi_h = gpuArray(phi_h);
    z_h = gpuArray(z_h);
    s_h = gpuArray(s_h);
    muh = gpuArray(muh);
    beta = gpuArray(beta);
    mask = gpuArray(single(mask));
end
catch
    disp('WARNING: GPU disabled');
end
    



E1t = conj(E1);
E2t = conj(E2);
E3t = conj(E3);

EE2 = E1t .* E1 + E2t .* E2 + E3t .* E3;
%Lap = EE2;

fprintf('%10s\t%10s\t%10s\t%10s\n', 'Outer Iter', 'Iter', 'Update', 'Delta');
%tic
delta = inf;
for t = 1:num_iter
    
   
    % update x : susceptibility estimate
    tx = E1t .* fftn(z_dx - s_dx);
    ty = E2t .* fftn(z_dy - s_dy);
    tz = E3t .* fftn(z_dz - s_dz);
    
    x_prev = x;
    Dt_kspace = conj(kernel) .* fftn(z2-s2-(phi_h));
    x = mask.*real(ifftn( (mu * (tx + ty + tz) + Dt_kspace) ./ (eps + mu2*abs(kernel).^2 + mu * EE2) ));

    clear tx ty tz
    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    fprintf('%10d\t%10d\t%10.4f\t%10.4g\n', t, 0, x_update, delta);
    
    if x_update < tolUpdate
        break
    end
    
    
    if t < num_iter
        % update z : gradient variable
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
        
        
        rhs_z2 = mu2*real(ifftn(kernel.*Fx)+s2 +phi_h  );
        z2 =  rhs_z2 ./ mu2 ;

        % Newton-Raphson method
        delta = inf;
        inn = 0;
        while (delta > tolDelta && inn < 50)
            inn = inn + 1;
            norm_old = norm(z2(:));
            
            update = ( W .* sin(z2 - phase) + mu2*z2 - rhs_z2 ) ./ ( W .* cos(z2 - phase) + mu2 );            
        
            z2 = z2 - update;     
            delta = norm(update(:)) / norm_old;
        end
        clear rhs_z2 update
        fprintf('%10d\t%10d\t%10.4f\t%10.4g\n', t, inn, x_update, delta);
        
               
        
        Fphi_h = (muh * conj(EE2).*fftn(z_h-s_h) + mu2*fftn(z2-s2) - mu2*kernel.*Fx) ./ (eps + mu2 + muh * EE2.*conj(EE2)) ;
        phi_h = real(ifftn(Fphi_h));
        
        z_h = muh*(real(ifftn(EE2.*Fphi_h))+s_h)./(muh+beta*mask);
        
        s2 = s2 + real(ifftn(kernel.*Fx)) - z2+phi_h;
        s_h = s_h + real(ifftn(EE2.*Fphi_h)) - z_h;
        
        clear Fx Fphi_h
    end
    
    
end
% Extract output values
out.time = toc;toc

out.x = gather(x);
out.phi = gather(phi_h);
out.iter = gather(t);



end
