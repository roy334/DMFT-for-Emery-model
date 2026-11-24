function [RhoV2, Delta, G, F] = emery_lattice_asymest(ocont, mu, P, SE, varargin)
% < Description >
%
% [RhoV2, Delta, G, F] = emery_lattice_asymest(ocont, mu, P, SE [,options])
%
% Single-site DMFT lattice step for the Emery model (oxygens integrated out)
% with the asymmetric estimator for the hybridization:
%
%   zeta(w) = w + i*eta + mu - ed - SE(w)
%   W(z,k)  = w (zO - h_pp)^{-1} w^dag,   zO = z - SEeta(w)   (causal O broadening)
%   G(w)    = (1/pi^2) int_[0,pi]^2 d^2k   1/(zeta - W(z,k))
%   F(w)    = (1/pi^2) int_[0,pi]^2 d^2k   W(z,k)/(zeta - W(z,k))
%   Delta   = F/G                    (= zeta - 1/G exactly; no subtraction of 1/G)
%   RhoV2   = -Im(Delta)/pi, clipped at 0
%
% z = w + i*eta + mu. There is no static term to subtract (t_cl = ed is
% already in zeta), unlike the one-band case where Delta = F/G - EKin.
% W/(zeta - W) stays bounded (-> -1) at the poles of W, so F is a smooth
% integrand.
%
% < Option >
% 'eta', ..    : imaginary shift (default 1e-4). Either a scalar (constant
%                broadening) or a vector with one value per frequency of ocont
%                (frequency-dependent broadening, e.g. eta(w) = max(eta0, a*|w|)).
%                eta acts as an artificial lifetime for ALL states; the oxygens
%                have no self-energy (U_p = 0), so their features have width ~eta.
% 'RelTol', .. : integral2 relative tolerance (default 1e-3)
% 'AbsTol', .. : integral2 absolute tolerance (default 1e-6)
% 'SEeta', ..  : causal O broadening on ocont (vector, numel(ocont)), added to the
%                oxygens only: zO = z - SEeta(w). [] or omitted = no O broadening.
%                Kramers-Kronig exact if SEeta is (e.g. the two-pole form).
%
% Requires emeryW.m. Same outputs as integrateG: RhoV2, Delta, and G.

eta = 1e-4;  RelTol = 1e-3;  AbsTol = 1e-6;  SEeta = [];
while ~isempty(varargin)
   switch varargin{1}
      case 'eta',    eta    = varargin{2}; varargin(1:2) = [];
      case 'RelTol', RelTol = varargin{2}; varargin(1:2) = [];
      case 'AbsTol', AbsTol = varargin{2}; varargin(1:2) = [];
      case 'SEeta',  SEeta  = varargin{2}; varargin(1:2) = [];
      otherwise, error(['ERR: unknown option ', num2str(varargin{1})]);
   end
end

ocont = ocont(:);  SE = SE(:);  No = numel(ocont);

% broadening per frequency: a scalar eta is expanded to all frequencies (previous
% behaviour); a vector eta gives a frequency-dependent broadening eta(w)
if isscalar(eta)
   eta = eta*ones(No,1);
else
   eta = eta(:);
   if numel(eta) ~= No
      error('ERR: ''eta'' must be a scalar or have numel(ocont) elements.');
   end
end

if isempty(SEeta), SEeta = zeros(No,1); end
SEeta = SEeta(:);
if numel(SEeta) ~= No
   error('ERR: ''SEeta'' must have numel(ocont) elements.');
end

G = zeros(No,1);  F = zeros(No,1);
parfor it = 1:No
   z    = ocont(it) + 1i*eta(it) + mu;     % lattice step evaluated at w + i*eta(w)
   zO   = z - SEeta(it);                   % oxygens: causal broadening SEeta(w)
   zeta = z - P.ed - SE(it);
   
   G(it) = integral2(@(kx,ky) 1./(zeta - emeryW(kx,ky,z,P,zO)), 0,pi,0,pi, ...
                     'AbsTol',AbsTol,'RelTol',RelTol,'Method','iterated')/pi^2;
   F(it) = integral2(@(kx,ky) Wfrac(kx,ky,z,zO,zeta,P), 0,pi,0,pi, ...
                     'AbsTol',AbsTol,'RelTol',RelTol,'Method','iterated')/pi^2;


end
Delta = F./G;
RhoV2 = max(-imag(Delta)/pi, 0);
end

function y = Wfrac(kx,ky,z,zO,zeta,P)
W = emeryW(kx,ky,z,P,zO);
y = W./(zeta - W);
end
