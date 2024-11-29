function [ phi0, deltaB, inderr ] = ilinearLSQR( Phase, TE, weight )
%This function performs a linear least squared error fit to multiecho GRE phase data.
%Two stages are performed, where the phase data with the largest discrepancy is ignored
%in the second run.
%Please remember to use unwrapped phase data!
%
% Required fields:
% Phase: all echoes phase container (4th dimension echo index), in radians, and unwrapped
% TE: echo times, in ms.
% weight: weights data container (all echoes), rescaled to the [0,1] range. Magnitude or Magnitude*TE is suggested.
%
% Last modified by Carlos Milovic in 2020.10.20

tic
W = weight.*weight;
wp = W.*Phase;

N = size(Phase(:,:,:,1));
f1 = zeros(N,'double');
f2 = zeros(N,'double');

a11 = zeros(N,'double');
a22 = zeros(N,'double');
a12 = zeros(N,'double');

for c = 1:length(TE)
    a11 = a11 + W(:,:,:,c) ;
    a12 = a12 + TE(c)*W(:,:,:,c);    
    a22 = a22 + TE(c)*TE(c)*W(:,:,:,c);

    f1 = f1+wp(:,:,:,c);
    f2 = f2+TE(c)*wp(:,:,:,c);
end
dA = eps+a11.*a22 - a12.*a12;

phi0 =  (a22.*f1-a12.*f2)./dA ;
deltaB = (-a12.*f1+a11.*f2)./(dA);

%deltaB = -((a22.*f1-a21.*f2)./(dA));
%phi0 =  (a21.*f1+a11.*f2)./dA ;

error = zeros(size(Phase));
for i = 1:length(TE)
    y = phi0+TE(i)*deltaB;
    error(:,:,:,i) = (y-Phase(:,:,:,i)).^2;
end

[merr,inderr] = max(error,[],4);

for i=1:N(1)
    for j=1:N(2)
        for k=1:N(3)
W(i,j,k,inderr(i,j,k)) = 0.0;
        end
    end
end

wp = W.*Phase;

f1 = zeros(N,'double');
f2 = zeros(N,'double');

a11 = zeros(N,'double');
a22 = zeros(N,'double');
a12 = zeros(N,'double');

for c = 1:length(TE)
    a11 = a11 + W(:,:,:,c) ;
    a12 = a12 + TE(c)*W(:,:,:,c);    
    a22 = a22 + TE(c)*TE(c)*W(:,:,:,c);

    f1 = f1+wp(:,:,:,c);
    f2 = f2+TE(c)*wp(:,:,:,c);
end
dA = eps+a11.*a22 - a12.*a12;

phi0 =  (a22.*f1-a12.*f2)./dA ;
deltaB = (-a12.*f1+a11.*f2)./(dA);


toc

end
