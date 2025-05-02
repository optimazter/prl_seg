function [ kernel ] = dipole_kernel_angulated( N, spatial_res, B0_dir, mode )
% This function calculates the dipole kernel used in QSM (single orientation)
% that models the susceptibility-to-field convolution.
% Use this function for angulated acquisitions, i.e. the main field direction is not along the z axis.
% We recommend rotating the acquisitions back to B0 along the z axis instead of using this function
% for better results (Kiersnowski O, et al. ISMRM 2021, p0794).
%
% This function uses an extension to the continuous kernel proposed by Salomir, et al. 2003.
%
% Parameters:
% N: array size
% spatial_res: voxel size in mm.
% B0_dir: main field direction, e.g. [0 0 1]
%
% Output:
% kernel: dipole kernel in the frequency space
%
% Created by Carlos Milovic, 30.03.2017
% Modified by Julio Acosta-Cabronero, 26.05.2017
% Last Modified by Carlos Milovic, 06.07.2020
if nargin < 4
    mode = 1;
end
N = single(N);
spatial_res = spatial_res/max(spatial_res(:));

if mode==0
[ky,kx,kz] = (meshgrid(-floor(N(2)/2):ceil(N(2)/2)-1, -floor(N(1)/2):ceil(N(1)/2)-1, -floor(N(3)/2):ceil(N(3)/2)-1));

kx = (single(kx) / max(abs(single(kx(:))))) / spatial_res(1);
ky = (single(ky) / max(abs(single(ky(:))))) / spatial_res(2);
kz = (single(kz) / max(abs(single(kz(:))))) / spatial_res(3);

k2 = kx.^2 + ky.^2 + kz.^2;
k2(k2==0) = eps;

%R_tot = eye(3); % Original formulation with a rotation matrix
%kernel = ifftshift( 1/3 - (kx * R_tot(3,1) + ky * R_tot(3,2) + kz * R_tot(3,3)).^2 ./ k2 ); 

% JAC
kernel = ifftshift( 1/3 - (kx*B0_dir(1) + ky*B0_dir(2) + kz*B0_dir(3)).^2 ./ k2 );    
kernel(1,1,1) = 0.0;


elseif  mode==1
    [k1, k2, k3] = ndgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);
    %[k1,k2,k3] = meshgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);

E1 = (1 - exp(2i .* pi .* k1 / N(1))) / spatial_res(1);
E2 = (1 - exp(2i .* pi .* k2 / N(2))) / spatial_res(2);
E3 = (1 - exp(2i .* pi .* k3 / N(3))) / spatial_res(3);
Lap = E1.*conj(E1)+E2.*conj(E2)+E3.*conj(E3);

kernel = (E1*B0_dir(1) + E2*B0_dir(2) + E3*B0_dir(3)).*conj(E1*B0_dir(1) + E2*B0_dir(2) + E3*B0_dir(3));
kernel = 1/3 - kernel./Lap;
kernel(1,1,1) = 0.0;

elseif  mode==2
FOV = N.*spatial_res;
center = 1+floor(N/2);

kx = 1:N(1);
ky = 1:N(2);
kz = 1:N(3);

kx = kx - center(1);
ky = ky - center(2);
kz = kz - center(3);

delta_kx = N(1)/FOV(1);
delta_ky = N(2)/FOV(2);
delta_kz = N(3)/FOV(3);


kx = kx / delta_kx;
ky = ky / delta_ky;
kz = kz / delta_kz;

kx = reshape(kx,[length(kx),1,1]);
ky = reshape(ky,[1,length(ky),1]);
kz = reshape(kz,[1,1,length(kz)]);

kx = repmat(kx,[1,N(2),N(3)]);
ky = repmat(ky,[N(1),1,N(3)]);
kz = repmat(kz,[N(1),N(2),1]);

spatial_kernel_z = zeros(N);
spatial_kernel_x = zeros(N);
spatial_kernel_y = zeros(N);

x = [0 0];
y = [0 0];
z = [0 0];

for i=1:N(1)
    for j=1:N(2)
        for k=1:N(3)
            x(1) = kx(i,j,k)-0.5/delta_kx;
            x(2) = kx(i,j,k)+0.5/delta_kx;
            y(1) = ky(i,j,k)-0.5/delta_ky;
            y(2) = ky(i,j,k)+0.5/delta_ky;
            z(1) = kz(i,j,k)-0.5/delta_kz;
            z(2) = kz(i,j,k)+0.5/delta_kz;
            
            m_z = zeros([2 2 2]);
            m_x = zeros([2 2 2]);
            m_y = zeros([2 2 2]);
            for di = 1:2
                for dj = 1:2
                    for dk = 1:2
                        m_z(di,dj,dk) = (-1)^(di+dj+dk) * (atan(x(di)*y(dj)/(z(dk)*sqrt(x(di)^2+y(dj)^2+z(dk)^2)))*B0_dir(3) - asinh(y(dj)/(sqrt(x(di)^2+z(dk)^2)))*B0_dir(1) - asinh(x(di)/(sqrt(y(dj)^2+z(dk)^2)))*B0_dir(2));
                        m_x(di,dj,dk) = (-1)^(di+dj+dk) * (atan(z(dk)*y(dj)/(x(di)*sqrt(x(di)^2+y(dj)^2+z(dk)^2)))*B0_dir(1) - asinh(y(dj)/(sqrt(x(di)^2+z(dk)^2)))*B0_dir(3) - asinh(z(dk)/(sqrt(y(dj)^2+x(di)^2)))*B0_dir(2));
                        m_y(di,dj,dk) = (-1)^(di+dj+dk) * (atan(z(dk)*x(di)/(y(dj)*sqrt(x(di)^2+y(dj)^2+z(dk)^2)))*B0_dir(2) - asinh(z(dk)/(sqrt(x(di)^2+y(dj)^2)))*B0_dir(1) - asinh(x(di)/(sqrt(y(dj)^2+z(dk)^2)))*B0_dir(3));
                    end
                end
            end
            spatial_kernel_z(i,j,k) = -(1/(pi*4)) *sum(m_z(:));       
            spatial_kernel_x(i,j,k) = -(1/(pi*4)) *sum(m_x(:));  
            spatial_kernel_y(i,j,k) = -(1/(pi*4)) *sum(m_y(:));       
            
            
        end
    end
end

DC = (kx==0) & (ky==0) & (kz==0);

spatial_kernel = (B0_dir(1)*spatial_kernel_x + B0_dir(2)*spatial_kernel_y + B0_dir(3)*spatial_kernel_z);

spatial_kernel(DC==1) = spatial_kernel(DC==1) + 1/3;

kernel = fftn(ifftshift(spatial_kernel));
%kernel.x = spatial_kernel_x;
%kernel.y = spatial_kernel_y;
%kernel.z = spatial_kernel_z;

elseif  mode==3

FOV = N.*spatial_res;
center = 1+floor(N/2);

kx = 1:N(1);
ky = 1:N(2);
kz = 1:N(3);

kx = kx - center(1);
ky = ky - center(2);
kz = kz - center(3);


% determine the step sizes delta_kx, delta_ky, delta_kz in mm
delta_kx = N(1)/FOV(1);
delta_ky = N(2)/FOV(2);
delta_kz = N(3)/FOV(3);


kx = kx / delta_kx;
ky = ky / delta_ky;
kz = kz / delta_kz;

kx = reshape(kx,[length(kx),1,1]);
ky = reshape(ky,[1,length(ky),1]);
kz = reshape(kz,[1,1,length(kz)]);

kx = repmat(kx,[1,N(2),N(3)]);
ky = repmat(ky,[N(1),1,N(3)]);
kz = repmat(kz,[N(1),N(2),1]);

%spatial_kernel = zeros(N);
r2 = kx.*kx+ky.*ky+kz.*kz;
spatial_kernel = (3*(kx*B0_dir(1)+ky*B0_dir(2)+kz*B0_dir(3)).^2-r2)./( 4*pi*(r2.^(5/2)));
spatial_kernel(r2==0) = 0.0;


%DC = (kx==0) & (ky==0) & (kz==0);
%spatial_kernel(DC==1) = spatial_kernel(DC==1)+1/3;

kernel = real(fftn(ifftshift(spatial_kernel)));
kernel(1,1,1) = 0.0;

end

end

