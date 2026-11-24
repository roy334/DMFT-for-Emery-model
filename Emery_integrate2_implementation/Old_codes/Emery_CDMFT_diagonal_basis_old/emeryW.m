function [W,wA2w,trAinv] = emeryW(kx,ky,z,P)
% Oxygen-mediated Cu-Cu kernel W(z,k) = w (z - h_pp(k))^{-1} w^dag of the Emery
% model (real gauge for h_pp; W, wA2w, trAinv are gauge invariant).
% Vectorized in kx, ky.  z = omega + i*eta + mu.
% Oxygen diagonal: ep - 2 tpp + 2 tpp2 cos k  (convention of Sordi et al.)
sx = sin(kx/2);  sy = sin(ky/2);
xx = P.ep - 2*P.tpp + 2*P.tpp2*cos(kx);
xy = P.ep - 2*P.tpp + 2*P.tpp2*cos(ky);
h  = 4*P.tpp*sx.*sy;
D  = (z-xx).*(z-xy) - h.^2;
b1 = ((z-xy).*sx + h.*sy)./D;
b2 = (h.*sx + (z-xx).*sy)./D;
W  = 4*P.tpd^2*(sx.*b1 + sy.*b2);
if nargout > 1
   wA2w   = 4*P.tpd^2*(b1.^2 + b2.^2);   % = -dW/dz
   trAinv = (2*z - xx - xy)./D;
end
end
