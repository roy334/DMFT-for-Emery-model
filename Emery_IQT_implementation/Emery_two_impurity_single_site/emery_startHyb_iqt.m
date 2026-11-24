function [Rho, SE0, n0, RhoAll] = emery_startHyb_iqt(ocont, mu, P, U, T, startHyb, orb, varargin)
% < Description >
%
% [Rho, SE0, n0, RhoAll] = emery_startHyb(ocont, mu, P, U, T, startHyb, orb [,options])
%
% Starting hybridization of ONE impurity (Cu or O) of single-site DMFT for the
% Emery model. Single-site code (U_p = 0): call once with orb = 'd'.
% Two-impurity code: call once with orb = 'd' and once with orb = 'p', passing
% the self-energy of the other orbital with 'SEother'.
%
% Starting self-energy of orbital a (level e_d = ed, e_p = ep - 2tpp):
%   'metallic' : Sigma_a = 0, lattice step at mu0 (default mu)
%   'hubbardI' : Sigma_a = z - e_a - 1/G_at,
%                G_at = (1 - n/2)/(z - e_a) + (n/2)/(z - e_a - U),  z = w + i*etaHI + mu
%   'hartree'  : Sigma_a = U*n/2 - i*Gamma0
% For 'hubbardI' and 'hartree', n (both spins; per O orbital for orb = 'p') is
% determined self-consistently at fixed mu (linear mixing 0.5):
%   n = 2 int dw f(w) (-Im G_a(w)/pi),  G_a from the lattice step.
% Lattice step: emery_lattice_twoimp.m with Sigma = diag(SEd, SEp, SEp), where the
% orbital not being started gets 'SEother' (default 0). For orb = 'd' and
% SEother = 0 this is identical to emery_lattice_asymest.m.
%
% < Input >
% ocont    : real frequencies (measured from mu)
% mu       : chemical potential
% P        : struct with ed, ep, tpd, tpp, tpp2
% U        : interaction of this orbital (Ud for 'd', Up for 'p')
% T        : temperature
% startHyb : 'metallic', 'hubbardI' or 'hartree'
% orb      : 'd' (Cu) or 'p' (O)
%
% < Option >
% 'eta', ..     : lattice broadening, scalar or numel(ocont) values (default 0.02)
% 'SEother', .. : self-energy of the other orbital on ocont (default 0)
% 'mu0', ..     : chemical potential for 'metallic' (default mu)
% 'etaHI', ..   : broadening inside the Hubbard-I atomic G (default 0.05)
% 'Gamma0', ..  : constant broadening for 'hartree' (default 0.5)
% 'n0', ..      : starting guess of n (default 1 for 'd', 1.9 for 'p')
% 'Nit', ..     : maximum number of occupation iterations (default 30)
% 'tol', ..     : tolerance on n (default 1e-3)
%
% < Output >
% Rho    : starting Gamma of this orbital (real, >= 0)
% SE0    : starting self-energy of this orbital
% n0     : occupation used in SE0 (NaN for 'metallic')
% RhoAll : [Gamma_d, Gamma_p] of the final lattice step (both columns)

ocont = ocont(:);  No = numel(ocont);
eta = 0.02;  SEother = zeros(No,1);  mu0 = mu;  etaHI = 0.05;  Gamma0 = 0.5;
n0 = [];  Nit = 30;  tol = 1e-3;
while ~isempty(varargin)
   switch varargin{1}
      case 'eta',     eta     = varargin{2}; varargin(1:2) = [];
      case 'SEother', SEother = varargin{2}; varargin(1:2) = [];
      case 'mu0',     mu0     = varargin{2}; varargin(1:2) = [];
      case 'etaHI',   etaHI   = varargin{2}; varargin(1:2) = [];
      case 'Gamma0',  Gamma0  = varargin{2}; varargin(1:2) = [];
      case 'n0',      n0      = varargin{2}; varargin(1:2) = [];
      case 'Nit',     Nit     = varargin{2}; varargin(1:2) = [];
      case 'tol',     tol     = varargin{2}; varargin(1:2) = [];
      otherwise, error(['ERR: unknown option ', num2str(varargin{1})]);
   end
end
if isscalar(SEother), SEother = SEother*ones(No,1); end
SEother = SEother(:);

switch orb
   case 'd', e = P.ed;            io = 1;  if isempty(n0), n0 = 1;   end
   case 'p', e = P.ep - 2*P.tpp;  io = 2;  if isempty(n0), n0 = 1.9; end
   otherwise, error('ERR: orb must be ''d'' or ''p''.');
end

% lattice step with this orbital's SE and the other orbital's SEother
latt = @(SE,m) emery_lattice_iqt(ocont,m,P, ...
          pick(io==1,SE,SEother), pick(io==2,SE,SEother), 'eta',eta);

z  = ocont + 1i*etaHI + mu;
fw = 1./(exp(ocont/T) + 1);          % ocont is measured from mu

switch startHyb
   case 'metallic'
      SE0 = zeros(No,1);
      [RhoAll,~,~] = latt(SE0,mu0);
      n0 = NaN;

   case {'hubbardI','hartree'}
      for it = 1:Nit
         if strcmp(startHyb,'hubbardI')
            Gat = (1-n0/2)./(z-e) + (n0/2)./(z-e-U);
            SE0 = z - e - 1./Gat;
         else
            SE0 = (U*n0/2 - 1i*Gamma0)*ones(No,1);
         end
         [RhoAll,~,Gl] = latt(SE0,mu);
         n_new = 2*trapz(ocont, fw.*(-imag(Gl(:,io))/pi));
         fprintf('%s seed (%s): it %2d  n_in = %.4f  n_out = %.4f\n', startHyb, orb, it, n0, n_new);
         if abs(n_new - n0) < tol, break; end
         n0 = 0.5*n0 + 0.5*n_new;
      end
      if abs(n_new - n0) >= tol
         warning('%s seed (%s): n not converged (last change %.2e)', startHyb, orb, abs(n_new - n0));
      end

   otherwise
      error('ERR: startHyb must be ''metallic'', ''hubbardI'' or ''hartree''.');
end

RhoAll = real(RhoAll);
Rho    = RhoAll(:,io);
end

function x = pick(c, a, b)
if c, x = a; else, x = b; end
end
