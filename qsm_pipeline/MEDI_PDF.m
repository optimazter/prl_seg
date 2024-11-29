function MEDI_PDF(input_T2_phase_path, input_mask_path, output_rdf_path, voxel_size, pdf_tolerance, TE)

    addpath(genpath(pwd))


    %%-------------------------------------------------------------------------
    %% MEDI STEP
    %%-------------------------------------------------------------------------

    [iField] = niftiread(input_T2_phase_path);
    [info] = niftiinfo(input_T2_phase_path);

    %Add Echo dimension if not already there. For Single-Echo
    if ndims(iField) < 4; 
        iField(end, end, end, 1) = 0; 
    end

    dim = info.raw.dim;
    matrix_size = [dim(2), dim(3), dim(4), 1];

    disp("Read matrix size");
    disp(matrix_size);

    
    disp("Read voxel size");
    disp(voxel_size);


    %Calculate quaternion
    quatern_b = info.raw.quatern_b;
    quatern_c = info.raw.quatern_c;
    quatern_d = info.raw.quatern_d;
    quatern_a = sqrt(1.0 - (quatern_b^2 + quatern_c^2 + quatern_d^2 ));

    disp("Read Quaternion from NIfTI");
    disp([quatern_a, quatern_b, quatern_c, quatern_d]);

    rotmat = quat_to_rot(quatern_a, quatern_b, quatern_c, quatern_d);

    %assuming B0 is in the z-direction
    B0_dir = [0;0;1];
    B0 = rotmat \ B0_dir;
    disp("Calculated B0");
    disp(B0);

    %[iField_corrected] = iField_correction(iField, voxel_size, mask);
    [iFreq_raw, N_std] = Fit_ppm_complex(iField);

    disp("Calculated N_std");


    disp("Calculating magnitude.");
    iMag = sqrt(sum(abs(iField).^2, 4));


    %disp("Unwrapping phase");
    %iFreq = unwrapPhase(iMag, iFreq_raw, matrix_size);

    mask = niftiread(input_mask_path);


    % mask = BET(iMag,matrix_size,voxel_size);
    %niftiwrite(mask, input_mask_path);

    disp("Running PDF with tolerance");
    disp(pdf_tolerance);
    [RDF shim] = PDF(iFreq_raw, N_std, mask, matrix_size, voxel_size, B0, pdf_tolerance);
    
    %Save the the cropped background dipole distribution as well as the local field map"
    niftiwrite(RDF, output_rdf_path);


end

