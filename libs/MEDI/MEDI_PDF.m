function MEDI_PDF(input_path, input_mask_path, output_path)

    addpath("/home/adrian-hjertholm-voldseth/dev/MEDI_toolbox/functions")

    iField = niftiread(input_path);
    mask = niftiread(input_mask_path);
    info = niftiinfo(input_path);

    dim = info.raw.dim;
    matrix_size = [dim(1), dim(2), dim(3)];
    
    pix_dim = info.raw.pixdim;
    voxel_size = [pix_dim(1), pix_dim(2), pix_dim(3)];

    %Calculate quaternion
    quatern_b = info.raw.quatern_b;
    quatern_c = info.raw.quatern_c;
    quatern_d = info.raw.quatern_d;
    quatern_a = sqrt(1.0 - (quatern_b^2 + quatern_c^2 + quatern_d^2 ));

    %Transform from quaternion representation to rotation matrix

    Rxx = 1 - 2*(quatern_c^2 + quatern_d^2);
    Rxy = 2*(quatern_b*quatern_c - quatern_d*quatern_a);
    Rxz = 2*(quatern_b*quatern_d + quatern_c*quatern_a);

    Ryx = 2*(quatern_b*quatern_c + quatern_d*quatern_a);
    Ryy = 1 - 2*(quatern_b^2 + quatern_d^2);
    Ryz = 2*(quatern_c*quatern_d - quatern_b*quatern_a );

    Rzx = 2*(quatern_b*quatern_d - quatern_c*quatern_a );
    Rzy = 2*(quatern_c*quatern_d + quatern_b*quatern_a );
    Rzz = 1 - 2 *(quatern_b^2 + quatern_c^2);

    rotmat = [ 
        Rxx,    Rxy,    Rxz;
        Ryx,    Ryy,    Ryz;
        Rzx,    Rzy,    Rzz];

    %assuming B0 is in the quatern_d-direction
    B0_dir = [0;0;1];
    B0 = rotmat \ B0_dir;

    %[iField_corrected] = iField_correction(iField, voxel_size, mask);
    [iFreq_raw, N_std] = Fit_ppm_complex(iField);
    pdf = PDF(iFreq_raw, N_std, mask, matrix_size, voxel_size, B0);

    niftiwrite(pdf, output_path);

end