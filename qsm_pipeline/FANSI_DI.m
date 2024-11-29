function FANSI_DI(input_rdf_path, input_mask_path, output_di_path, alpha1, mu1, phs_scale, spatial_res)
    %%-------------------------------------------------------------------------
    %% FANSI STEP
    %%-------------------------------------------------------------------------


    addpath(genpath(pwd))


    [phs] = niftiread(input_rdf_path);
    [mask] = niftiread(input_mask_path);
    iMag = sqrt(sum(abs(phs).^2,4));


    
    %%-------------------------------------------------------------------------
    %% Nonlinear TV
    %%-------------------------------------------------------------------------
    N = size(mask);
    kernel = dipole_kernel_fansi( N, spatial_res, 0 );
    
    params = [];
    params.input = phs;
    mag_use = iMag .* mask;
    mag_use = mag_use / max(abs(mag_use(:)));
    params.weight = mag_use;
    params.K = kernel;

    params.alpha1 = alpha1;  % gradient L1 penalty
    params.mu1 = mu1;        % gradient consistency

    outw = wTV(params);
    chiw = outw.x/phs_scale;

    niftiwrite(chiw, output_di_path);

end