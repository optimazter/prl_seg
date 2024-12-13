function [ e1, e2 ] = compute_freqe_auto( chi, m1, m2 )
% Calculates the Mean Amplitude of the energy inside masks
% m1, m2 and (optional) m3 in the Fourier domain of QSM reconstruction chi.
%
% The estimation of the Mean Amplitude may be the mean magnitude 
% of the coefficients (power = 1) or the squared mean of the 
% coefficients (power = 2). Both are robust estimators.
%
% Last modified by Carlos Milovic in 2020.07.08
%

fchi = (abs(fftn(chi)));

n1 = sum(m1(:));
n2 = sum(m2(:));

e1 = sum ( fchi(:).*m1(:) )/n1;
e2 = sum ( fchi(:).*m2(:) )/n2;

end
