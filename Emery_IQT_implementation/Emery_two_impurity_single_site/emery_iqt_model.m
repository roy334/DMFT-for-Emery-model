function [q, eloc] = emery_iqt_model(P)
% < Description >
%
% [q, eloc] = emery_iqt_model(P)
%
% IQT denominator model of the Emery lattice in the cell-periodic gauge
% (all orbitals of a cell share the phase e^{ik.R_i}; Eq. (10) of the notes),
% basis (d, p_x, p_y), without mu (mu enters through z):
%
%   H(k) = [ ed                  tpd(1-e^{-ikx})                 tpd(1-e^{-iky})               ]
%          [ tpd(1-e^{ikx})      xi_x                            tpp(1-e^{ikx})(1-e^{-iky})    ]
%          [ tpd(1-e^{iky})      tpp(1-e^{-ikx})(1-e^{iky})      xi_y                          ]
%
%   xi_x = ep - 2tpp + 2tpp2 cos kx,   xi_y = ep - 2tpp + 2tpp2 cos ky
%
% as a Fourier series H(k) = sum_m C_m exp(i(m1 ky + m2 kx)), modes [m1 m2] =
% [outer = ky, inner = kx] (IQT's MATLAB order), period 2*pi in both.
%
% < Input >
% P : struct with ed, ep, tpd, tpp, tpp2 (ep already includes any epDC shift)
%
% < Output >
% q    : iqt.resolvent of H, i.e. the denominator z - H(k) (Sigma is passed per solve)
% eloc : [ed, ep-2tpp, ep-2tpp], the impurity levels (k-average of the diagonal)

t = P.tpd;  s = P.tpp;  t2 = P.tpp2;  ept = P.ep - 2*P.tpp;
modes = [0 0; 0 1; 0 -1; 1 0; -1 0; -1 1; 1 -1];
C = zeros(3,3,7);
C(:,:,1) = [P.ed  t  t;  t ept  s;  t  s ept];   % constant parts
C(:,:,2) = [0 0 0; -t t2 -s; 0 0 0];             % e^{+i kx}
C(:,:,3) = C(:,:,2)';                            % e^{-i kx}
C(:,:,4) = [0 0 0; 0 0 0; -t -s t2];             % e^{+i ky}
C(:,:,5) = C(:,:,4)';                            % e^{-i ky}
C(:,:,6) = [0 0 0; 0 0 s; 0 0 0];                % e^{i(kx-ky)}
C(:,:,7) = C(:,:,6)';                            % e^{-i(kx-ky)}
H = iqt.fourier2d(modes, C);
if ~iqt.hermitian(H)
   error('ERR: Emery model is not Hermitian (check the coefficients).');
end
q = iqt.resolvent(H);
eloc = [P.ed, ept, ept];
end
