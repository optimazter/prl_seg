%http://martinos.org/~berkin/software.html
function QSM(input_unwrapped_phase, input_bet_mask, output_lbv_path, output_qsm_path, TE, B0, gyro, spatial_res)

    
    addpath(genpath(pwd))

    iFreq = niftiread(input_unwrapped_phase);
    N = size(iFreq);
    disp(N);

    mask = niftiread(input_bet_mask);


    % R2* map needed for ventricular CSF mask
    R2s = arlo(TE, abs(iField));

    % Ventricular CSF mask for zero referencing 
    %	Requirement:
    %		R2s:	R2* map
    Mask_CSF = extract_CSF(R2s, mask, voxel_size);


    % Morphology enabled dipole inversion with zero reference using CSF (MEDI+0)
    QSM = MEDI_L1('lambda', 1000, 'lambda_CSF', 100, 'merit', 'smv', 5);


    niftiwrite(QSM, output_qsm_path);

end

    