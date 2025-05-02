function [mssim, ssim_map] = compute_xsim(img1, img2, mask, sw)
% This function reimplements the SSIM metric for QSM comparisons.
% Please use this function instead of "compute_ssim"
%
% Changes made to the 2016 QSM Reconstruction Challenge implementation:
% - QSM maps are no longer rescaled. Keeping the reference and target in the 
% native [ppm] is recommended.
% - L parameter is set to 1 [ppm] accordingly.
% - K = [0.01 0.001] to promote detection of streaking artifacts.
% - mask parameter allows to set a custom ROI for evaluation.
% - sw parameter sets the size of the gaussian kernel (default=[3 3 3])
%
% See README.txt for more details.
%
% Modified by Carlos Milovic in 2019.05.26
%
% New version 2.0 (Jan 2022) includes a complete rewriting of the 
% function for optimization purposes. Apart from a new argument parser,
% the new version uses imgaussfilt3 instead of convn, which is faster.
% Custom calculation of the gauss filter is deprecated. This new version 
% is also compatible with gpuArrays, if used as input.
%
% Last modified by Carlos Milovic in 2022.01.26
%
% See below part of the original header of this function:
%========================================================================
%SSIM Index, Version 1.0
%Copyright(c) 2003 Zhou Wang
%All Rights Reserved.
%
%The author was with Howard Hughes Medical Institute, and Laboratory
%for Computational Vision at Center for Neural Science and Courant
%Institute of Mathematical Sciences, New York University, USA. He is
%currently with Department of Electrical and Computer Engineering,
%University of Waterloo, Canada.
%
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for calculating the
%Structural SIMilarity (SSIM) index between two images. Please refer
%to the following paper:
%
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%quality assessment: From error measurement to structural similarity"
%IEEE Transactios on Image Processing, vol. 13, no. 4, Apr. 2004.
%
%Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%            *** Note that in XSIM the index map size is the same as the 
%            input images ***
%========================================================================

% Initial function parsing check
if (nargin < 2 || nargin > 4)
   disp('Failed: Incorrect number of arguments!');
   mssim = -Inf;
   ssim_map = -Inf;
   return;
end

if (size(img1) ~= size(img2))
   disp('Failed: Reference and Target images do not have matching dimensions!');
   mssim = -Inf;
   ssim_map = -Inf;
   return;
end

s = size(img1);



%--------------------------------------------------------------------------

% Argument parsing

if (nargin < 4)  
   sw = [3 3 3];
end
if ((s(1) < sw(1)) || (s(2) < sw(2)) || (s(3) < sw(3)))
   disp('Failed: Gaussian window is larger than the image!');
   mssim = -Inf;
   ssim_map = -Inf;
   return
end
swt = 2*sw+1;
%window = gkernel(1.5,sw);	%
%window = window/sum(window(:));

if (nargin == 2)  
    mask = img2 ~=0;
end

K(1) = 0.01;								      % default settings
K(2) = 0.001;								      %
L = 1;                                  % Please see Milovic et al, ISMRM'22 for details
% if (length(K) == 2)
%     if (K(1) < 0 || K(2) < 0)
%         disp('Failed: K cannot have negative values!')
%         mssim = -Inf;
%         ssim_map = -Inf;
%         return;
%     end
% else
%     disp('Failed: Wrong number of K elements!')
%     mssim = -Inf;
%     ssim_map = -Inf;
%     return;
% end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;


%img1 = double(img1);
%img2 = double(img2);

mu1   = imgaussfilt3(img1,1.5,'FilterSize',swt);
mu2   = imgaussfilt3(img2,1.5,'FilterSize',swt);

mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;

sigma1_sq = imgaussfilt3(img1.*img1,1.5,'FilterSize',swt) - mu1_sq;
sigma2_sq = imgaussfilt3(img2.*img2,1.5,'FilterSize',swt) - mu2_sq;
sigma12 = imgaussfilt3(img1.*img2,1.5,'FilterSize',swt) - mu1_mu2;


ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));

mssim = sum(ssim_map(:).*mask(:))/sum(mask(:));

return

%function [gaussKernel] = gkernel(sigma,sk)
% Rewriten function by Carlos Milovic

%[ky,kx,kz] = meshgrid(-sk(2):sk(2), -sk(1):sk(1), -sk(3):sk(3));
%k2 = kx.^2 + ky.^2 + kz.^2;
%gaussKernel = exp(-k2/(2*sigma^2));
