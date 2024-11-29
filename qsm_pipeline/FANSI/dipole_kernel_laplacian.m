function [ kernel, lap ] = dipole_kernel_laplacian( N, spatial_res )
% This function calculates a discrete wave dipole kernel formulation used in
% single step QSM methods (Langkammer et al, MRM 2015, 10.1016/j.neuroimage.2015.02.041)
% Use this function if the main field direction is along the z axis.
% In addition to the dipole kernel, the discrete laplacian operator is provided.
%
% Parameters:
% N: array size
% spatial_res: voxel size in mm.
%
% Output:
% kernel: dipole kernel in the frequency space
% lap: laplacian in the frequency space
%
% Last Modified by Carlos Milovic, 26.02.2021



    
FOV = N.*spatial_res;
center = 1+floor(N/2);
kx = 1:N(1);
ky = 1:N(2);
kz = 1:N(3);

kx = kx - center(1);
ky = ky - center(2);
kz = kz - center(3);

delta_kx = 1/FOV(1);
delta_ky = 1/FOV(2);
delta_kz = 1/FOV(3);


kx = kx * delta_kx;
ky = ky * delta_ky;
kz = kz * delta_kz;

kx = reshape(kx,[length(kx),1,1]);
ky = reshape(ky,[1,length(ky),1]);
kz = reshape(kz,[1,1,length(kz)]);

kx = repmat(kx,[1,N(2),N(3)]);
ky = repmat(ky,[N(1),1,N(3)]);
kz = repmat(kz,[N(1),N(2),1]);

kernel = (2/3)*(cos(2*pi*kx)+cos(2*pi*ky)-2*cos(2*pi*kz));

kernel = ifftshift(kernel);  % Keep the center of the frequency domain at [1,1,1]
kernel(1,1,1) = 0.0;
    

lap = 2*(cos(2*pi*kx)+cos(2*pi*ky)+cos(2*pi*kz)-3);

lap = ifftshift(lap);  % Keep the center of the frequency domain at [1,1,1]
lap(1,1,1) = 0.0;


end
