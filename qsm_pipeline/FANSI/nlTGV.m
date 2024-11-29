function out = nlTGV(params,varargin)
% Nonlinear QSM and Total Generalized Variation regularization 
% with spatially variable fidelity and regularization weights.
% This uses ADMM to solve the functional.
%
% Parameters: params - structure with 
% Required fields:
% params.input: local field map
% params.alpha1: gradient penalty (L1-norm) or regularization weight
% Optional fields:
% params.K: dipole kernel in the frequency space
% params.mu1: gradient consistency weight (ADMM weight, recommended = 100*alpha1)
% params.mu2: fidelity consistency weight (ADMM weight, recommended value = 1.0)
% params.alpha0: curvature L1 penalty, regularization weight (recommended = 2*alpha1)
% params.mu0: curvature consistency weight (ADMM weight, recommended = 2*mu1)
% params.maxOuterIter: maximum number of ADMM iterations (recommended = 150)
% params.tolUpdate: convergence limit, change rate in the solution (recommended = 0.1)
% params.weight: data fidelity spatially variable weight (recommended = magnitude_data). 
% params.regweight - regularization spatially variable weight.
% params.isPrecond: preconditionate solution by smart initialization (default = true)
% params.isGPU: GPU acceleration (default = true)
%
% Output: out - structure with the following fields:
% out.x: calculated susceptibility map
% out.iter: number of iterations needed
% out.time: elapsed time (excluding pre-calculations)
% out.totalTime: total elapsed time (including pre-calculations)
%
% Based on the code by Bilgic Berkin at http://martinos.org/~berkin/software.html
% Modified by Carlos Milovic in 2017.03.31
% Modified by Carlos Milovic in 2020.07.07
% Last modified by Carlos Milovic and Patrich Fuchs in 2021.10.11

totalTime = tic;

[phase, alpha1, alpha0, N, Kernel, mu1, mu0, mu2, maxOuterIter, ...
 tolUpdate, regweight, tolDelta, W, isGPU,isPrecond, isVoutput] = parse_inputs(params, varargin{:});
    
    
    
    % Precompute gradient-related matrices
    [k1, k2, k3] = ndgrid(0:N(1)-1, 0:N(2)-1, 0:N(3)-1);
    E1 = 1 - exp(2i .* pi .* k1 / N(1));
    E2 = 1 - exp(2i .* pi .* k2 / N(2));
    E3 = 1 - exp(2i .* pi .* k3 / N(3)); 

    
    
    if isPrecond
        z2 =  W.*phase./(W+mu2);
    else
        z2 = zeros(N,'single');
    end
    s2 = zeros(N,'single'); 
    
    % Allocate memory for first order gradient
    s1_1 = zeros(N,'single'); z1_1 = zeros(N,'single');
    s1_2 = zeros(N,'single'); z1_2 = zeros(N,'single');
    s1_3 = zeros(N,'single'); z1_3 = zeros(N,'single');
    
    % Allocate memory for symmetrized gradient
    s0_1 = zeros(N,'single'); z0_1 = zeros(N,'single'); 
    s0_2 = zeros(N,'single'); z0_2 = zeros(N,'single');
    s0_3 = zeros(N,'single'); z0_3 = zeros(N,'single'); 
    s0_4 = zeros(N,'single'); z0_4 = zeros(N,'single');
    s0_5 = zeros(N,'single'); z0_5 = zeros(N,'single');
    s0_6 = zeros(N,'single'); z0_6 = zeros(N,'single');

    x = zeros(N, 'single');
    
    
% Move variables to GPU
try
if isGPU 
    
    disp('GPU enabled!');
    phase = gpuArray(phase);

    z1_1 = gpuArray(z1_1);
    z1_2 = gpuArray(z1_2);
    z1_3 = gpuArray(z1_3);

    z0_1 = gpuArray(z0_1);
    z0_2 = gpuArray(z0_2);
    z0_3 = gpuArray(z0_3);
    z0_4 = gpuArray(z0_4);
    z0_5 = gpuArray(z0_5);
    z0_6 = gpuArray(z0_6);

    s1_1 = gpuArray(s1_1);
    s1_2 = gpuArray(s1_2);
    s1_3 = gpuArray(s1_3);
    
    s0_1 = gpuArray(s0_1);
    s0_2 = gpuArray(s0_2);
    s0_3 = gpuArray(s0_3);
    s0_4 = gpuArray(s0_4);
    s0_5 = gpuArray(s0_5);
    s0_6 = gpuArray(s0_6);


    x = gpuArray(x);
    Kernel = gpuArray(Kernel);

    z2 = gpuArray(z2);
    s2 = gpuArray(s2);

    tolUpdate = gpuArray(tolUpdate);

    E1 = gpuArray(E1);
    E2 = gpuArray(E2);
    E3 = gpuArray(E3);

    regweight = gpuArray(regweight);
    alpha1 = gpuArray(alpha1);
    alpha0 = gpuArray(alpha0);
    mu1 = gpuArray(mu1);
    mu2 = gpuArray(mu2);
    mu0 = gpuArray(mu0);
    W = gpuArray(W);
end
catch
    disp('WARNING: GPU disabled');
end
    
    
    
    Kt = conj(Kernel);
    
    Et1 = conj(E1);     Et2 = conj(E2);     Et3 = conj(E3);
    
    E1tE1 = Et1.*E1;    E2tE2 = Et2.*E2;    E3tE3 = Et3.*E3;
    mu0_over_2_E1tE2 = mu0/2*Et1.*E2;
    mu0_over_2_E1tE3 = mu0/2*Et1.*E3;
    mu0_over_2_E2tE3 = mu0/2*Et2.*E3;

    a0 = mu2*Kt.*Kernel;
    a0_mu1_E_sos    = a0 + mu1*(E1tE1 + E2tE2 + E3tE3);
    mu1I_mu0_E_wsos1 = mu1 + mu0*(E1tE1 + (E2tE2 + E3tE3)/2);
    mu1I_mu0_E_wsos2 = mu1 + mu0*(E1tE1/2 + E2tE2 + E3tE3/2);
    mu1I_mu0_E_wsos3 = mu1 + mu0*((E1tE1 + E2tE2)/2 + E3tE3);
    
    clear E1tE1 E2tE2 E3tE3 
    
    %% Precomputation for Cramer's Rule 
    a1 = a0_mu1_E_sos; a2 = mu1I_mu0_E_wsos1;  a3 = mu1I_mu0_E_wsos2;  a4 = mu1I_mu0_E_wsos3;
    a5 = -mu1*E1;       a6 = -mu1*E2;           a7 = mu0_over_2_E1tE2;  a8 = -mu1*E3;   a9 = mu0_over_2_E1tE3;  a10 = mu0_over_2_E2tE3;
    a5t = conj(a5);     a6t = conj(a6);         a7t = conj(a7);         a8t = conj(a8); a9t = conj(a9);         a10t = conj(a10);    
    
    clear a0_mu1_E_sos mu1I_mu0_E_wsos1 mu1I_mu0_E_wsos2 mu1I_mu0_E_wsos3;
    
    % For x
    D11 = a2.*a3.*a4    + a7t.*a9.*a10t + a7.*a9t.*a10  - a3.*a9.*a9t   - a2.*a10.*a10t     - a4.*a7.*a7t;
    D21 = a3.*a4.*a5t   + a6t.*a9.*a10t + a7.*a8t.*a10  - a3.*a8t.*a9   - a5t.*a10.*a10t    - a4.*a6t.*a7;
    D31 = a4.*a5t.*a7t  + a6t.*a9.*a9t  + a2.*a8t.*a10  - a7t.*a8t.*a9  - a5t.*a9t.*a10     - a2.*a4.*a6t;
    D41 = a5t.*a7t.*a10t + a6t.*a7.*a9t + a2.*a3.*a8t   - a7.*a7t.*a8t  - a3.*a5t.*a9t      - a2.*a6t.*a10t;

    % For vx
    D12 = a3.*a4.*a5    + a7t.*a8.*a10t + a6.*a9t.*a10  - a3.*a8.*a9t   - a5.*a10.*a10t - a4.*a6.*a7t;
    D22 = a1.*a3.*a4    + a6t.*a8.*a10t + a6.*a8t.*a10  - a3.*a8.*a8t   - a1.*a10.*a10t - a4.*a6.*a6t;
    D32 = a1.*a4.*a7t   + a6t.*a8.*a9t  + a5.*a8t.*a10  - a7t.*a8.*a8t  - a1.*a9t.*a10  - a4.*a5.*a6t;
    D42 = a1.*a7t.*a10t + a6.*a6t.*a9t  + a3.*a5.*a8t   - a6.*a7t.*a8t  - a1.*a3.*a9t   - a5.*a6t.*a10t;

    % For vy
    D13 = a4.*a5.*a7 + a2.*a8.*a10t + a6.*a9.*a9t - a7.*a8.*a9t - a5.*a9.*a10t - a2.*a4.*a6;
    D23 = a1.*a4.*a7 + a5t.*a8.*a10t +a6.*a8t.*a9 - a7.*a8.*a8t - a1.*a9.*a10t - a4.*a5t.*a6;
    D33 = a1.*a2.*a4 + a5t.*a8.*a9t + a5.*a8t.*a9 - a2.*a8.*a8t - a1.*a9.*a9t - a4.*a5.*a5t;
    D43 = a1.*a2.*a10t + a5t.*a6.*a9t + a5.*a7.*a8t - a2.*a6.*a8t - a1.*a7.*a9t - a5.*a5t.*a10t;

    % For vz
    D14 = a5.*a7.*a10 + a2.*a3.*a8 + a6.*a7t.*a9 - a7.*a7t.*a8 - a3.*a5.*a9 -a2.*a6.*a10;
    D24 = a1.*a7.*a10 + a3.*a5t.*a8 + a6.*a6t.*a9 - a6t.*a7.*a8 - a1.*a3.*a9 - a5t.*a6.*a10;
    D34 = a1.*a2.*a10 + a5t.*a7t.*a8 + a5.*a6t.*a9 - a2.*a6t.*a8 - a1.*a7t.*a9 - a5.*a5t.*a10;
    D44 = a1.*a2.*a3 + a5t.*a6.*a7t + a5.*a6t.*a7 - a2.*a6.*a6t - a1.*a7.*a7t - a3.*a5.*a5t;

    det_A = a1.*D11 - a5.*D21 + a6.*D31 - a8.*D41;
    det_Ainv = 1 ./ (eps+det_A);
    
    clear mu0_over_2_E1tE2 mu0_over_2_E1tE3 mu0_over_2_E2tE3 a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a6t a7t a8t a9t a10t det_A;
    
    fprintf('%3s\t%10s\n', 'Iter', 'Update');
    tic
    for t = 1:maxOuterIter
        
        % Update x and v
        F_z0_minus_s0_1 = fftn(z0_1 - s0_1);
        F_z0_minus_s0_2 = fftn(z0_2 - s0_2);
        F_z0_minus_s0_3 = fftn(z0_3 - s0_3);
        F_z0_minus_s0_4 = fftn(z0_4 - s0_4);
        F_z0_minus_s0_5 = fftn(z0_5 - s0_5);
        F_z0_minus_s0_6 = fftn(z0_6 - s0_6);
        

        F_z1_minus_s1_1 = fftn(z1_1 - s1_1);
        F_z1_minus_s1_2 = fftn(z1_2 - s1_2);
        F_z1_minus_s1_3 = fftn(z1_3 - s1_3);
        
        
        rhs1    =  mu2*(Kt.*fftn( z2-s2 )) + mu1*(Et1.*F_z1_minus_s1_1 + Et2.*F_z1_minus_s1_2 + Et3.*F_z1_minus_s1_3);
        rhs2    = -mu1*F_z1_minus_s1_1 + mu0*(Et1.*F_z0_minus_s0_1 + Et2.*F_z0_minus_s0_4 + Et3.*F_z0_minus_s0_5);
        rhs3    = -mu1*F_z1_minus_s1_2 + mu0*(Et2.*F_z0_minus_s0_2 + Et1.*F_z0_minus_s0_4 + Et3.*F_z0_minus_s0_6);
        rhs4    = -mu1*F_z1_minus_s1_3 + mu0*(Et3.*F_z0_minus_s0_3 + Et1.*F_z0_minus_s0_5 + Et2.*F_z0_minus_s0_6);
        
        clear F_z0_minus_s0_1 F_z0_minus_s0_2 F_z0_minus_s0_3 F_z0_minus_s0_4 F_z0_minus_s0_5 F_z0_minus_s0_6 F_z1_minus_s1_1 F_z1_minus_s1_2 F_z1_minus_s1_3;
        
        % Cramer's rule
        Fx = (rhs1.*D11 - rhs2.*D21 + rhs3.*D31 - rhs4.*D41) .* det_Ainv;
        Fv1 = (-rhs1.*D12 + rhs2.*D22 - rhs3.*D32 + rhs4.*D42) .* det_Ainv;
        Fv2 = (rhs1.*D13 - rhs2.*D23 + rhs3.*D33 - rhs4.*D43) .* det_Ainv;
        Fv3 = (-rhs1.*D14 +rhs2.*D24 - rhs3.*D34 + rhs4.*D44) .* det_Ainv;       
            
        clear rhs1 rhs2 rhs3 rhs4
        
        x_prev = x;
        x = real(ifftn(Fx));
        v1 = real(ifftn(Fv1));
        v2 = real(ifftn(Fv2));
        v3 = real(ifftn(Fv3));
        
        x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
        fprintf('%3d\t%10.4f\n', t, x_update);
    
        if x_update < tolUpdate
            break
        end
        
        Fx = fftn(x); % Extra step to increase stability from the proximal step
        Fv1 = fftn(v1);
        Fv2 = fftn(v2);
        Fv3 = fftn(v3);
        
        % Compute gradients for z0 and z1 update
        Dx1 = real(ifftn(E1.*Fx));
        Dx2 = real(ifftn(E2.*Fx));
        Dx3 = real(ifftn(E3.*Fx));

        E_v1 = real(ifftn(E1.*Fv1));
        E_v2 = real(ifftn(E2.*Fv2));
        E_v3 = real(ifftn(E3.*Fv3));
        E_v4 = real(ifftn(E1.*Fv2 + E2.*Fv1))/2;
        E_v5 = real(ifftn(E1.*Fv3 + E3.*Fv1))/2;
        E_v6 = real(ifftn(E2.*Fv3 + E3.*Fv2))/2;
        
        clear Fv1 Fv2 Fv3;
        
        % Update z0: Symm grad
        z0_1 = max(abs(E_v1 + s0_1)-alpha0/mu0,0).*sign(E_v1 + s0_1);
        z0_2 = max(abs(E_v2 + s0_2)-alpha0/mu0,0).*sign(E_v2 + s0_2);
        z0_3 = max(abs(E_v3 + s0_3)-alpha0/mu0,0).*sign(E_v3 + s0_3);
        z0_4 = max(abs(E_v4 + s0_4)-alpha0/mu0,0).*sign(E_v4 + s0_4);
        z0_5 = max(abs(E_v5 + s0_5)-alpha0/mu0,0).*sign(E_v5 + s0_5);
        z0_6 = max(abs(E_v6 + s0_6)-alpha0/mu0,0).*sign(E_v6 + s0_6);
        
        % Update z1: Grad
        z1_1 = max(abs(Dx1-v1+s1_1)-regweight(:,:,:,1)*alpha1/mu1,0).*sign(Dx1-v1+s1_1);
        z1_2 = max(abs(Dx2-v2+s1_2)-regweight(:,:,:,2)*alpha1/mu1,0).*sign(Dx2-v2+s1_2);
        z1_3 = max(abs(Dx3-v3+s1_3)-regweight(:,:,:,3)*alpha1/mu1,0).*sign(Dx3-v3+s1_3);

        rhs_z2 = mu2*real(ifftn(Kernel.*Fx)+s2  );
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
%         if DEBUG; disp(delta); end   
                
         
        % Update s0 and s1
        s0_1 = s0_1 + E_v1-z0_1;
        s0_2 = s0_2 + E_v2-z0_2;
        s0_3 = s0_3 + E_v3-z0_3;
        s0_4 = s0_4 + E_v4-z0_4;
        s0_5 = s0_5 + E_v5-z0_5;
        s0_6 = s0_6 + E_v6-z0_6;
        
        s1_1 = s1_1 + Dx1-v1-z1_1;
        s1_2 = s1_2 + Dx2-v2-z1_2;
        s1_3 = s1_3 + Dx3-v3-z1_3;
        
        clear Dx1 Dx2 Dx3 E_v1 E_v2 E_v3 E_v4 E_v5 E_v6;
        
        s2 = s2 + real(ifftn(Kernel.*Fx)) - z2;
        
    end
    
out.time = toc;toc
out.totalTime = toc(totalTime);
if isGPU
    out.x = gather(x);
    if isVoutput   % Enable is you require the computation of the auxiliary variable v
        out.v1 = gather(v1); 
        out.v2 = gather(v2);
        out.v3 = gather(v3);
    end
else
    out.x = x;
    if isVoutput   % Enable is you require the computation of the auxiliary variable v
        out.v1 = v1; 
        out.v2 = v2;
        out.v3 = v3;
    end
end
out.iter = t;


end





function [phase, alpha1, alpha0, N, Kernel, mu1, mu0, mu2, maxOuterIter, tolUpdate, regweight, tolDelta, W, isGPU, isPrecond, isVoutput] = parse_inputs(params, varargin)

if isstruct(params)
    if isfield(params,'input')
        phase = params.input;
    else
        error('Please provide a struct with "input" field as the local phase input.')
    end
    if isfield(params,'alpha1')
        alpha1 = params.alpha1;
    else
        error('Please provide a struct with "alpha1" field as input.')
    end
else
    error('Please provide a struct with "input" and "alpha1" fields as input.')
end
N = size(phase);

% Define Defaults
defaultAlpha0 = 2*alpha1;
defaultMu1 = 100*alpha1;
defaultMu0 = 2*defaultMu1;
defaultMu2 = 1.0;
defaultNoOuter = 150;
defaultTol = 0.1;
defaultRegweight = ones([N 3]);
defaultDeltaTol = 1e-6;
defaultW = ones(N);

p = inputParser;
p.KeepUnmatched = true;
addRequired(p,  'phase',                         @(x) isnumeric(x));
addRequired(p,  'alpha1',                        @(x) isscalar(x));
addParameter(p, 'alpha0',       defaultAlpha0,   @(x) isscalar(x));
addParameter(p, 'K',            [],              @(x) isnumeric(x));
addParameter(p, 'mu1',          defaultMu1,      @(x) isscalar(x));
addParameter(p, 'mu0',          defaultMu0,      @(x) isscalar(x));
addParameter(p, 'mu2',          defaultMu2,      @(x) isscalar(x));
addParameter(p, 'maxOuterIter', defaultNoOuter,  @(x) isscalar(x));
addParameter(p, 'tolUpdate',    defaultTol,      @(x) isscalar(x));
addParameter(p, 'regweight',    defaultRegweight,@(x) isnumeric(x));
addParameter(p, 'tolDelta',     defaultDeltaTol, @(x) isscalar(x));
addParameter(p, 'weight',       defaultW,        @(x) isnumeric(x));
addParameter(p, 'magnitude',    [],              @(x) isnumeric(x));
addParameter(p, 'isPrecond',    true,            @(x) islogical(x));
addParameter(p, 'isGPU',        true,            @(x) islogical(x));
addParameter(p, 'isVoutput',    false,           @(x) islogical(x));

% if false; fprintf(1,'Parsing inputs...'); end
parse(p, phase, alpha1, params, varargin{:});

phase    = single(p.Results.phase);
alpha1   = single(p.Results.alpha1);

if any(strcmpi(p.UsingDefaults, 'k'))
    if isfield(params,'voxelSize')
        voxelSize = params.voxelSize;
    else
        voxelSize = [1,1,1];
    end
    if isfield(params,'B0direction')
        B0direction = params.B0direction;
    else
        B0direction = [0,0,1];
    end
    Kernel = dipole_kernel_angulated( N, voxelSize, B0direction ); 
else
    Kernel   = p.Results.K;
end

alpha0       = p.Results.alpha0;
mu1          = p.Results.mu1;
mu0          = p.Results.mu0;
mu2          = p.Results.mu2;
maxOuterIter = p.Results.maxOuterIter;
tolUpdate    = p.Results.tolUpdate;

regweight = p.Results.regweight;
if not(any(strcmpi(p.UsingDefaults, 'regweight')))
    if length(size(regweight)) == 3
            regweight = repmat(regweight,[1,1,1,3]);
    end
end

tolDelta = p.Results.tolDelta;
magnitude = p.Results.magnitude;

if any(strcmpi(p.UsingDefaults, 'w'))
    warning('No weight supplied.')
end
if any(strcmpi(p.UsingDefaults, 'w')) && not(any(strcmpi(p.UsingDefaults, 'magnitude')))
    W = magnitude.*magnitude;
else
    W = p.Results.weight.*p.Results.weight;
end

isGPU         = p.Results.isGPU;
isPrecond     = p.Results.isPrecond;

isVoutput     = p.Results.isVoutput;


% if DEBUG; fprintf(1,'\tDone.\n'); end
end
