function MEDI_PDF(input_path, input_mask_path, output_path, voxel_size_x, voxel_size_y, voxel_size_z)

    addpath("/home/adrian-hjertholm-voldseth/dev/MEDI_toolbox/functions")

    [iField_raw] = niftiread(input_path);
    [iField_unwrapped] = niftiread(input_unwrapped_path);
    [info] = niftiinfo(input_path);
    [mask] = niftiread(input_mask_path);

    %Add Echo dimension if not already there. For Single-Echo
    if ndims(iField_raw) < 4; iField(end, end, end, 1) = 0; end
    if ndims(iField_unwrapped) < 4; iField(end, end, end, 1) = 0; end

    dim = info.raw.dim;
    matrix_size = [dim(2), dim(3), dim(4), 1];

    disp("Reading matrix dimensions...");
    disp(matrix_size);
    

    voxel_size = [voxel_size_x, voxel_size_y, voxel_size_z];
    
    disp("Reading voxel size...");
    disp(voxel_size);


    %Calculate quaternion
    quatern_b = info.raw.quatern_b;
    quatern_c = info.raw.quatern_c;
    quatern_d = info.raw.quatern_d;
    quatern_a = sqrt(1.0 - (quatern_b^2 + quatern_c^2 + quatern_d^2 ));

    disp("Reading quaternion...");
    disp(quatern_a);
    disp(quatern_b);
    disp(quatern_c);
    disp(quatern_d);

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

    %assuming B0 is in the z-direction
    B0_dir = [0;0;1];
    B0 = rotmat \ B0_dir;

    %[iField_corrected] = iField_correction(iField, voxel_size, mask);
    [iFreq_raw, N_std] = Fit_ppm_complex(iField_raw);

    %   input to PDF:
    %   iFreq - the unwrapped field map
    %   N_std - the noise standard deviation on the field map. (1 over SNR for single echo)
    %   Mask - a binary 3D matrix denoting the Region Of Interest
    %   matrix_size - the size of the 3D matrix
    %   voxel_size - the size of the voxel in mm
    %   B0_dir - the direction of the B0 field
    %   tol(optional) - tolerance level
    