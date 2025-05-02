% This is a sample script to show how to use the functions in this toolbox, 
% and how to set the principal variables.
% This example uses an analytic brain phantom.
%
% Based on the code by Bilgic Berkin at http://martinos.org/~berkin/software.html
% Modified by Carlos Milovic in 2017.12.27
% Last modified by Carlos Milovic in 2020.07.13
%

set(0,'DefaultFigureWindowStyle','docked')

addpath(genpath(pwd)) % Please run this while at the FANSI root folder

%% load data

load chi_phantom
load mask_phantom
load spatial_res_phantom

N = size(chi);

imagesc3d2(chi - (mask_use==0), N/2, 1, [90,90,90], [-0.12,0.12], 0, 'True Susceptibility') 
 
center = N/2 + 1;


% Simulate magnitude data based on the susceptibility map

mag = chi-min(chi(:));
mag = mag/max(mag(:)).*mask_use;


% Add simulated lesions - Constant spheres (optional)
add_lessions = false;
if add_lessions == true
center1 = N/2;
[chiS, Bnuc] = chi_intsphere(spatial_res.*N,N,center1,6, -0.3, 0);
chi = chi+chiS;
mag = mag.*(0.0+1+chiS/0.3);

center2 = [59 86 75];
[chiS, Bnuc] = chi_intsphere(spatial_res.*N,N,center2,6, +0.6, 0);
chi = chi+chiS;
mag = mag.*(0.0+1-chiS/0.6);

center3 = [163 94 24];
[chiS, Bnuc] = chi_intsphere(spatial_res.*N,N,center3,6, +1.2, 0);
chi = chi+chiS;
mag = mag.*(0.0+1-chiS/1.2);

center4 = [105 171 57];
[chiS, Bnuc] = chi_intsphere(spatial_res.*N,N,center4,6, -0.5, 0);
chi = chi+chiS;
mag = mag.*(0.0+1+chiS/0.5);
end


%% Create dipole kernel and susceptibility to field model

kernel = dipole_kernel_fansi( N, spatial_res, 0 ); % 0 for the continuous kernel by Salomir and Marques and Bowtell.
                                             % 1 for a discrete kernel formulation
                                             % 2 for an integrated Green function

chi = chi-mean(chi(:)); % Demean the susceptibility ground-truth.
phase_true = real(ifftn(kernel .* fftn(chi))); % field map in ppm

% Simulate acquisition parameters
TE = 5e-3; %5ms
B0 = 3;
gyro = 2*pi*42.58;
phs_scale = TE * gyro * B0; % ppm-to-radian scaling factor

%% Add noise

SNR = 345; % peak SNR value
noise = 1/SNR;
signal = mag.*exp(1i*phase_true*phs_scale)+noise*(randn(N)+1i*randn(N));
phase_use = angle(signal)/phs_scale; % Let's work in ppm in this example.
                                     % Since we are simulating only a local field, and short TE,
                                     % the unwrapping step is skipped.
magn_use = abs(signal).*mask_use;

% Estimate the noise impact in the phase.
rmse_noise = compute_rmse(mask_use.*phase_use,mask_use.*phase_true); % Multiply by the mask to select the ROI


% Add phase 2pi jumps into the phase data to simulate unwrapping errors (optional)
add_jumps = false;
if add_jumps == true
pt = [152 102 center(3)];
phase_use(pt(1),pt(2),pt(3)) = phase_use(pt(1),pt(2),pt(3)) - 2*pi;
pt = [110 152 center(3)];
phase_use(pt(1),pt(2),pt(3)) = phase_use(pt(1),pt(2),pt(3)) + 2*pi;
pt = [113 118 center(3)];
phase_use(pt(1),pt(2),pt(3)) = phase_use(pt(1),pt(2),pt(3)) + 2*pi;
pt = [78 106 center(3)];
phase_use(pt(1),pt(2),pt(3)) = phase_use(pt(1),pt(2),pt(3)) - 2*pi;
pt = [133 183 center(3)];
phase_use(pt(1),pt(2),pt(3)) = phase_use(pt(1),pt(2),pt(3)) + 2*pi;
end

imagesc3d2(phase_use, N/2, 2, [90,90,90], [-0.4 0.4], 0, ['Noise RMSE: ', num2str(rmse_noise)])
imagesc3d2(magn_use, N/2, 3, [90,90,90], [0,1], 0, ['Noisy Magnitude']) 
 

%% TKD recon 

kthre = 0.08;       % truncation threshold
chi_tkd = tkd( phase_use.*mask_use, mask_use, kernel, kthre, N );

rmse_tkd = compute_rmse(chi_tkd.*mask_use, chi); % Calculate RMSE only
metrics_tkd = compute_metrics(real(chi_tkd.*mask_use),chi); % Calculate RMSE, HFEN, SSIM and XSIM.

imagesc3d2(chi_tkd .* mask_use - (mask_use==0), N/2, 4, [90,90,90], [-0.12,0.12], 0, ['TKD RMSE: ', num2str(rmse_tkd)])


%% Closed-form Tikhonov-Gradient Domain recon

beta = 3e-3;    % regularization parameter
chi_L2 = chiL2( phase_use.*mask_use, mask_use, kernel, beta, N );

rmse_L2 = compute_rmse(chi_L2.*mask_use, chi);
metrics_l2 = compute_metrics(real(chi_L2.*mask_use),chi);

imagesc3d2(chi_L2 .* mask_use - (mask_use==0), N/2, 5, [90,90,90], [-0.12,0.12], 0, ['L2 RMSE: ', num2str(rmse_L2)])



%% Linear TV and TGV ADMM recon
% For this example, we'll work with the input phase in ppm

% Required parameters
params = []; % reset the structure
params.K = kernel;
params.input = phase_use;

params.alpha1 = 2e-4;               % gradient L1 penalty

% Optional
params.weight = single(mask_use); % To avoid noise from external regions to corrupt the solution, 
                          % it is recommended to either mask this weight, or the input data.
%params.maxOuterIter = 50; % Use these values for faster reconstructions.
%params.tol_update = 1;

% TV regularization
out = wTV(params); 
rmse_tv = compute_rmse(out.x.*mask_use , chi);
metrics_tv = compute_metrics(out.x.*mask_use,chi);

params.alpha1 = 1e-4;  
params.weight = magn_use; % Compare it with a reconstruction using the magnitude as data fidelity weight.
                          % Note that the magnitude is normalized to the 0-1 range.
outw = wTV(params); 
rmse_tvw = compute_rmse(outw.x.*mask_use , chi);
metrics_tvw = compute_metrics(outw.x.*mask_use,chi);

% Display results
imagesc3d2(out.x .* mask_use - (mask_use==0), N/2, 6, [90,90,90], [-0.12,0.12], 0, ['TV RMSE: ', num2str(rmse_tv), '  iter : ', num2str(out.iter)])

imagesc3d2(outw.x .* mask_use - (mask_use==0), N/2, 7, [90,90,90], [-0.12,0.12], 0, ['Weighted TV RMSE: ', num2str(rmse_tvw), '  iter : ', num2str(outw.iter)])


% TGV
% By default, the second order penalty is automatically set to twice the first order penalty.
%params.alpha0 = 2 * params.alpha1;  % second order gradient L1 penalty

params.alpha1 = 2e-4;    
params.weight = mask_use; 
out2 = wTGV(params); 
rmse_tgv = 100 * norm(out2.x(:).*mask_use(:) - chi(:)) / norm(chi(:));
metrics_tgv = compute_metrics(out2.x.*mask_use,chi);

params.alpha1 = 1e-4;  
params.weight = magn_use;
out2w = wTGV(params); 
rmse_tgvw = compute_rmse(out2w.x.*mask_use , chi);
metrics_tgvw = compute_metrics(out2w.x.*mask_use,chi);

imagesc3d2(out2.x .* mask_use - (mask_use==0), N/2, 8, [90,90,90], [-0.12,0.12], 0, ['TGV RMSE: ', num2str(rmse_tgv), '  iter : ', num2str(out2.iter)])

imagesc3d2(out2w.x .* mask_use - (mask_use==0), N/2, 9, [90,90,90], [-0.12,0.12], 0, ['Weighted TGV RMSE: ', num2str(rmse_tgvw), '  iter : ', num2str(out2w.iter)])




%% Non linear TV and TGV ADMM recon
% For the nonlinear algorithms, we'll work with the input phase in radians.
% num_iter = 50;
% tol_update = 1;

% Required parameters
params = []; % reset the structure
params.K = kernel;
params.input = phase_use*phs_scale; % scale to radians

params.alpha1 = 2e-4;               % gradient L1 penalty

% Optional parameters
params.weight = magn_use; % Recommended for nonlinear algorithms
 
%params.maxOuterIter = 50; % Use these values for faster reconstructions.
%params.tol_update = 1;

% nlTV

outnl = nlTV(params); 
rmse_tvnl = compute_rmse(outnl.x.*mask_use/phs_scale,chi);
metrics_nltv = compute_metrics(outnl.x.*mask_use/phs_scale,chi);

imagesc3d2(outnl.x .* mask_use/phs_scale - (mask_use==0), N/2, 10, [90,90,90], [-0.12,0.12], 0, ['nlTV RMSE: ', num2str(rmse_tvnl), '  iter : ', num2str(outnl.iter)])


% nlTGV
%params.alpha0 = 2 * params.alpha1;  % second order gradient L1 penalty

out2nl = nlTGV(params); 
rmse_tgvnl = compute_rmse(out2nl.x.*mask_use/phs_scale,chi);
metrics_nltgv = compute_metrics(out2nl.x.*mask_use/phs_scale,chi);

imagesc3d2(out2nl.x .* mask_use/phs_scale - (mask_use==0), N/2, 11, [90,90,90], [-0.12,0.12], 0, ['nlTGV RMSE: ', num2str(rmse_tgvnl), '  iter : ', num2str(out2nl.iter)])



%% OTHER EXAMPLES OF USE
%% Spatially weighted regularization term (example with linear function) - Optional
% Use this to impose morphological similarity between the susceptibility map and the magnitude data
% (you may also use an R2* map for this purpose).

% Common parameters
params = [];

params.K = kernel;
params.input = phase_use;

params.alpha1 = 2e-4;               % gradient L1 penalty
params.weight = single(mask_use);

Gm = gradient_calc(magn_use,0); % 0 for vectorized gradient. Use options 1 or 2 to use the L1 or L2 of the gradient.

Gm = max(Gm,noise); % For continous weighting. Soft-threshold to avoid zero divisions and control noise.
params.regweight = mean(Gm(:))./Gm;

outrw = wTV(params); 
rmse_tvrw = 100 * norm(outrw.x(:).*mask_use(:) - chi(:)) / norm(chi(:));
metrics_tvrw = compute_metrics(real(outrw.x.*mask_use),chi);


bGm = threshold_gradient( Gm, mask_use, noise, 0.3 ); % For binary weighting. 
params.regweight = bGm;

outbrw = wTV(params); 
rmse_tvbrw = compute_rmse(real(outbrw.x.*mask_use),chi);%100 * norm(outbrw.x(:).*mask_use(:) - chi(:)) / norm(chi(:));
metrics_tvbrw = compute_metrics(real(outbrw.x.*mask_use),chi);

%% FANSI main function call example
% This is a wildcard to call linear or nonlinear methods, with TV or TGV as regularizer.
% The dipole kernel and weights are automatically calculated and selected.
% For parameter fine-tunning purposes is more efficient to call the individual functions as
% demonstrated in the previous sections of this script.
% *** Warning *** This function is not compatible with older than Release 3.0 code (Oct 2021).

options = [];
alpha1 = 1e-4;               % gradient L1 penalty
options.isNonlinear = false;
options.voxelSize = spatial_res;
options.noise = noise;
options.isGPU = true;
outf = FANSI( phase_use, magn_use, alpha1, options );
rmse_f1 = 100 * norm(outf.x(:).*mask_use(:) - chi(:)) / norm(chi(:));
 [ data_cost1, reg_cost1 ] = compute_costs( outf.x, phase_use, kernel );

imagesc3d2(real(outf.x) .* mask_use - (mask_use==0), N/2, 12, [90,90,90], [-0.12,0.12], 0, ['FANSI-TV RMSE: ', num2str(rmse_f1), '  iter : ', num2str(outf.iter)])

options.isTGV = true;
options.isNonlinear = true;
options.voxelSize = spatial_res;
options.noise = noise;
options.isGPU = false;
outf2 = FANSI( phase_use*phs_scale, magn_use, alpha1*2, options );
rmse_f2 = 100 * norm(outf2.x(:).*mask_use(:)/phs_scale - chi(:)) / norm(chi(:));
 [ data_cost2, reg_cost2 ] = compute_costs( outf2.x/phs_scale, phase_use, kernel );

imagesc3d2(real(outf2.x) .* mask_use/phs_scale - (mask_use==0), N/2, 13, [90,90,90], [-0.12,0.12], 0, ['FANSI-TGV RMSE: ', num2str(rmse_f2), '  iter : ', num2str(outf2.iter)])
