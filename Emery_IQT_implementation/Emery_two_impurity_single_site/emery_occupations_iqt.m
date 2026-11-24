function [nd, npo, ntot, Ad, Ap, info] = emery_occupations_iqt(w, mu, P, SEd, SEp, T, varargin)
% < Description >
%
% [nd, npo, ntot, Ad, Ap, info] = emery_occupations_iqt(w, mu, P, SEd, SEp, T [,options])
%
% Replacement of emery_occupations_twoimp.m using the IQT integrator.
% Occupations WITHOUT a frequency grid or broadening: iqt.expect2d integrates
% the Fermi function and the momentum together, off the real axis,
%
%   n_a = 2 * (1/4pi^2) int d^2k int dw f(w - mu) A_aa(k,w)     (both spins)
%
% with Sigma(z) continued from the data on the grid w by
% iqt.selfenergy_continuation (spectral part -Im Sigma/pi linear between the
% grid points, each interval in closed form). Model: emery_iqt_model.m.
%
%   nd   : Cu, both spins
%   npo  : ONE oxygen orbital, both spins
%   ntot = nd + 2*npo
%
% Ad, Ap (only computed if requested) are -Im G_a/pi on the grid w at
% z = w + i*eta + mu, from one iqt.Quad2D sweep.
%
% < Input >
% w     : real frequencies (increasing), measured from mu; e.g. ocont of the loop
% mu, P : chemical potential, model struct (as in emery_lattice_iqt.m)
% SEd   : Cu self-energy on w;  SEp : O self-energy on w (scalar 0 for U_p = 0)
% T     : temperature
%
% < Option >
% 'RelTol', ..  : relative tolerance of the occupations (default 1e-6)
% 'eta', ..     : broadening for Ad, Ap only (default 0.02)
% 'RelTolA', .. : relative tolerance for Ad, Ap (default 1e-5)
% 'Threads', .. : IQT worker threads (default 'auto')
%
% < Output >
% info : struct: statsN (expectation statistics), statsA (spectral sweep,
%        if computed), SigmaInf = [Re Sigma_d(inf), Re Sigma_p(inf)] estimated by
%        the continuation (compare with the Hartree values U n/2)

RelTol = 1e-6;  eta = 0.02;  RelTolA = 1e-5;  Threads = 'auto';
while ~isempty(varargin)
   switch varargin{1}
      case 'RelTol',  RelTol  = varargin{2}; varargin(1:2) = [];
      case 'eta',     eta     = varargin{2}; varargin(1:2) = [];
      case 'RelTolA', RelTolA = varargin{2}; varargin(1:2) = [];
      case 'Threads', Threads = varargin{2}; varargin(1:2) = [];
      otherwise, error(['ERR: unknown option ', num2str(varargin{1})]);
   end
end

w = w(:);  SEd = SEd(:);  SEp = SEp(:);  No = numel(w);
if isscalar(SEp), SEp = SEp*ones(No,1); end

[q, ~] = emery_iqt_model(P);
area = 4*pi^2;

% Sigma(z) off the axis; data are measured from mu, hence z - mu
[sd, sdInf] = iqt.selfenergy_continuation(w, SEd);
[sp, spInf] = iqt.selfenergy_continuation(w, SEp);
sigma = @(z) diag3(sd(z - mu), sp(z - mu));

[rho, statsN] = iqt.expect2d(iqt.constant(1,2), q, mu, T, [-pi pi], [-pi pi], ...
                 SelfEnergy=sigma, MatrixEntries=[1 1; 2 2], AbsTol=0, RelTol=RelTol, ...
                 Threads=Threads);
nd   = 2*real(rho(1))/area;
npo  = 2*real(rho(2))/area;
ntot = nd + 2*npo;

info.statsN   = statsN;
info.SigmaInf = real([sdInf, spInf]);
if ~all([statsN.Converged])
   warning('emery_occupations_iqt: expectation value did not converge (retcode %s).', ...
           string(statsN(1).Retcode));
end

if nargout > 3
   S = zeros(3,3,No);
   S(1,1,:) = SEd;  S(2,2,:) = SEp;  S(3,3,:) = SEp;
   cache = iqt.Quad2D(iqt.constant(1,2), q, MatrixEntries=[1 1; 2 2], ...
                      AbsTol=1e-8*area, RelTol=RelTolA, Threads=Threads);
   cleanup = onCleanup(@() delete(cache));
   z = w + mu + 1i*eta;
   [Gl, statsA] = cache.sweep(z.', [-pi pi], [-pi pi], 'SelfEnergy', S);
   Gl = Gl/area;
   Ad = -imag(Gl(1,:)).'/pi;
   Ap = -imag(Gl(2,:)).'/pi;
   info.statsA = statsA;
end
end

function S = diag3(a, b)
% a, b: 1-by-1-by-M  ->  3-by-3-by-M with diag(a, b, b)
M = size(a,3);
S = zeros(3,3,M);
S(1,1,:) = a;  S(2,2,:) = b;  S(3,3,:) = b;
end
