function [RhoV2, Delta, G, F, info] = emery_lattice_iqt(ocont, mu, P, SEd, SEp, varargin)
% < Description >
%
% [RhoV2, Delta, G, F, info] = emery_lattice_iqt(ocont, mu, P, SEd, SEp [,options])
%
% Drop-in replacement of emery_lattice_twoimp.m (same inputs, same first four
% outputs), with the k-integration done by the IQT integrator (iqt.Quad2D)
% instead of integral2. Model: emery_iqt_model.m (cell-periodic gauge, Eq. (10)).
%
%   Sigma(w) = diag(SEd(w), SEp(w), SEp(w)),   z = w + i*eta + mu
%   R(k,z)   = [z - H(k) - Sigma]^{-1}
%   G_a = <R_aa>_k ,   F_a = <[h R]_aa>_k ,   h = H - diag(ed, ep-2tpp, ep-2tpp)
%   Delta_a = F_a/G_a  (asymmetric estimator),   RhoV2_a = max(-Im Delta_a/pi, 0)
%   <.>_k = (1/4pi^2) int_[-pi,pi]^2 d^2k
%
% G and F come from ONE adaptive mesh per frequency (matrix numerator [1; h]):
% IQT integrates the fitted ratio analytically, so small eta is fine.
%
% < Input >
% ocont : real frequencies, measured from mu
% mu    : chemical potential
% P     : struct with ed, ep, tpd, tpp, tpp2
% SEd   : Cu self-energy on ocont
% SEp   : O self-energy on ocont (zeros or scalar 0 for U_p = 0)
%
% < Option >
% 'eta', ..     : broadening, scalar or one value per frequency (default 1e-4)
% 'RelTol', ..  : relative tolerance (default 1e-5)
% 'AbsTol', ..  : absolute tolerance on the k-AVERAGED quantities (default 1e-8)
% 'Threads', .. : IQT worker threads (default 'auto'; results do not depend on it)
%
% < Output >
% RhoV2 : [numel(ocont) x 2], columns: Gamma_d, Gamma_p
% Delta : [numel(ocont) x 2], columns: Delta_d, Delta_p
% G     : [numel(ocont) x 2], columns: G_d, G_p (local, per spin, per O orbital)
% F     : [numel(ocont) x 2], columns: F_d, F_p
% info  : struct: stats (IQT statistics per frequency), nFailed (frequencies
%         that neither converged nor ended 'roundoff'), nRoundoff, and
%         idErr = max_w |F_a - (zeta_a G_a - 1)| / max(|F_a|), the residual of
%         the exact identity F_a = zeta_a G_a - 1 (zero for exact integrals;
%         a direct measure of the integration error)

eta = 1e-4;  RelTol = 1e-5;  AbsTol = 1e-8;  Threads = 'auto';
while ~isempty(varargin)
   switch varargin{1}
      case 'eta',     eta     = varargin{2}; varargin(1:2) = [];
      case 'RelTol',  RelTol  = varargin{2}; varargin(1:2) = [];
      case 'AbsTol',  AbsTol  = varargin{2}; varargin(1:2) = [];
      case 'Threads', Threads = varargin{2}; varargin(1:2) = [];
      otherwise, error(['ERR: unknown option ', num2str(varargin{1})]);
   end
end

ocont = ocont(:);  SEd = SEd(:);  SEp = SEp(:);  No = numel(ocont);
if isscalar(SEp), SEp = SEp*ones(No,1); end
if numel(SEd) ~= No || numel(SEp) ~= No
   error('ERR: SEd and SEp must have numel(ocont) elements.');
end
if isscalar(eta), eta = eta*ones(No,1); else, eta = eta(:); end
if numel(eta) ~= No
   error('ERR: ''eta'' must be a scalar or have numel(ocont) elements.');
end

[q, eloc] = emery_iqt_model(P);
area = 4*pi^2;                                   % raw IQT integrals -> zone averages

% numerator [1; h]: rows 1-3 give R, rows 4-6 give h*R
[Hs, ~] = iqt.symbols;
h = Hs - diag(eloc);
cache = iqt.Quad2D([1; h], q, MatrixEntries=[1 1; 2 2; 4 1; 5 2], ...
                   AbsTol=AbsTol*area, RelTol=RelTol, Threads=Threads);
cleanup = onCleanup(@() delete(cache));

S = zeros(3,3,No);
S(1,1,:) = SEd;  S(2,2,:) = SEp;  S(3,3,:) = SEp;
z = ocont + mu + 1i*eta;
[V, stats] = cache.sweep(z.', [-pi pi], [-pi pi], 'SelfEnergy', S);
V = V/area;                                      % rows: G_d, G_p, F_d, F_p

G     = V(1:2,:).';
F     = V(3:4,:).';
Delta = F./G;
RhoV2 = max(-imag(Delta)/pi, 0);

% diagnostics
rc = arrayfun(@(s) string(s.Retcode), stats);
info.stats     = stats;
info.nRoundoff = sum(rc == "roundoff");
info.nFailed   = sum(~[stats.Converged] & rc ~= "roundoff");
zeta  = [z - eloc(1) - SEd, z - eloc(2) - SEp];
info.idErr = max(abs(F - (zeta.*G - 1)), [], 1) ./ max(abs(F), [], 1);
if info.nFailed > 0
   warning('emery_lattice_iqt: %d of %d frequencies did not converge (retcodes: %s).', ...
           info.nFailed, No, strjoin(unique(rc(~[stats.Converged] & rc ~= "roundoff")), ', '));
end
end
