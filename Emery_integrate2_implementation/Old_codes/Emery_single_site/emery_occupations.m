function [nd, np, ntot, Ad, Ap] = emery_occupations(ocont, mu, P, SE, T, varargin)
% < Description >
%
% [nd, np, ntot, Ad, Ap] = emery_occupations(ocont, mu, P, SE, T [,options])
%
% Occupations of the Emery model in single-site DMFT from the local
% self-energy SE(w) on the Cu d orbital (ocont measured from mu):
%
%   G_dd(k,w)   = 1/(z - ed - W(z,k) - SE(w)),        z = w + i*eta + mu
%   tr G_pp(k,w)= tr (z - h_pp)^{-1} + G_dd(k,w) * w (z - h_pp)^{-2} w^dag
%
%   nd   = 2 int dw f(w) A_d(w),     A_d = -Im <G_dd>_k / pi
%   np   = 2 int dw f(w) A_p(w),     A_p = -Im <tr G_pp>_k / (2 pi)   (per O orbital)
%   ntot = nd + 2*np
%
% The oxygen part is exact (U_p = 0): the oxygens are integrated out, and
% their Green's function follows from G_dd.
%
% < Option >
% 'eta', ..    : broadening (default 0.02). The free-oxygen part has poles on
%                the real axis, so a finite eta is needed for integral2.
%                Scalar (constant) or a vector with one value per frequency of
%                ocont (frequency-dependent; use the same eta(w) as the DMFT loop
%                so that the occupations belong to the converged spectra).
% 'RelTol', .. : integral2 relative tolerance (default 1e-4)
%
% Requires emeryW.m.

eta = 0.02;  RelTol = 1e-4;
while ~isempty(varargin)
   switch varargin{1}
      case 'eta',    eta    = varargin{2}; varargin(1:2) = [];
      case 'RelTol', RelTol = varargin{2}; varargin(1:2) = [];
      otherwise, error(['ERR: unknown option ', num2str(varargin{1})]);
   end
end

ocont = ocont(:);  SE = SE(:);  No = numel(ocont);

% broadening per frequency (scalar -> constant, vector -> eta(w))
if isscalar(eta)
   eta = eta*ones(No,1);
else
   eta = eta(:);
   if numel(eta) ~= No
      error('ERR: ''eta'' must be a scalar or have numel(ocont) elements.');
   end
end
Gd = zeros(No,1);  Gp = zeros(No,1);
parfor it = 1:No
   z = ocont(it) + 1i*eta(it) + mu;
   s = SE(it);
   Gd(it) = integral2(@(kx,ky) gdd(kx,ky,z,s,P), 0,pi,0,pi,'RelTol',RelTol)/pi^2;
   Gp(it) = integral2(@(kx,ky) trgpp(kx,ky,z,s,P), 0,pi,0,pi,'RelTol',RelTol)/pi^2;
end
Ad = -imag(Gd)/pi;
Ap = -imag(Gp)/(2*pi);
fw = 1./(exp(ocont/T)+1);
nd = 2*trapz(ocont, fw.*Ad);
np = 2*trapz(ocont, fw.*Ap);
ntot = nd + 2*np;
end

function g = gdd(kx,ky,z,s,P)
W = emeryW(kx,ky,z,P);
g = 1./(z - P.ed - W - s);
end

function t = trgpp(kx,ky,z,s,P)
[W,wA2w,trAinv] = emeryW(kx,ky,z,P);
t = trAinv + wA2w./(z - P.ed - W - s);
end
