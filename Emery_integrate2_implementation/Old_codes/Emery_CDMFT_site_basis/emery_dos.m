function [omega, Ad, Ap, Atot] = emery_dos(tpd, tpp, tpp2, ep, ed, varargin)
% < Description >
%
%   [omega, Ad, Ap, Atot] = emery_dos(tpd, tpp, tpp2, ep, ed, ...)
%
% Non-interacting density of states (per spin, per unit cell) of the
% three-band Emery model, orbitally resolved, computed with integral2.
%
%   A_d(w)  = -(1/pi) Im (1/N) sum_k [ (w + i*eta - h0(k))^{-1} ]_dd
%   A_p(w)  = -(1/pi) Im (1/N) sum_k [ (w + i*eta - h0(k))^{-1} ]_{px,px}
%             (= the p_y DOS by symmetry, so the total is Atot = Ad + 2*Ap)
%
% The oxygens are integrated out analytically:
%   G_dd(k)   = 1/(z - ed - W(z,k)),           W = w (z - h_pp)^{-1} w^dag
%   G_pxpx(k) = [(z - h_pp)^{-1}]_xx + G_dd(k) * [ (z-h_pp)^{-1} w^dag w (z-h_pp)^{-1} ]_xx
% evaluated in a gauge where h_pp is real (all quantities used are gauge
% invariant). The integrands are even in kx and ky, so the BZ integral is
% done over [0,pi]^2 with prefactor 1/pi^2.
%
% < Input >
%   tpd, tpp, tpp2 : Cu-O hopping, O-O hopping, longer-range O-O hopping (t'_pp)
%   ep, ed         : oxygen and copper on-site energies
%
% < Options >
%   'omega', ..    : frequency grid (default: linspace over the band range +- 2)
%   'eta', ..      : broadening (default 0.05). integral2 needs eta > 0; for
%                    eta << bandwidth/N_omega the integrand becomes very sharp.
%   'shift', ..    : true (default) -> oxygen diagonal is ep - 2 tpp + 2 tpp2 cos k
%                    (convention of Sordi et al., PRB 111, 045117).
%                    false -> ep + 2 tpp2 cos k (bare on-site energy ep).
%   'RelTol', 'AbsTol' : integral2 tolerances (default 1e-6, 1e-8)
%   '-noplot'      : do not plot
%
% < Output >
%   omega : frequency grid,  Ad : Cu d DOS,  Ap : DOS of ONE oxygen orbital,
%   Atot  : Ad + 2*Ap.   Sum rules: int Ad = 1, int Ap = 1, int Atot = 3.

% ----- options -----
eta = 0.05; omega = []; shift = true; RelTol = 1e-6; AbsTol = 1e-8; doplot = true;
while ~isempty(varargin)
   switch varargin{1}
      case 'omega',  omega  = varargin{2}; varargin(1:2) = [];
      case 'eta',    eta    = varargin{2}; varargin(1:2) = [];
      case 'shift',  shift  = varargin{2}; varargin(1:2) = [];
      case 'RelTol', RelTol = varargin{2}; varargin(1:2) = [];
      case 'AbsTol', AbsTol = varargin{2}; varargin(1:2) = [];
      case '-noplot', doplot = false;      varargin(1)   = [];
      otherwise, error(['ERR: unknown option ', num2str(varargin{1})]);
   end
end

P = struct('tpd',tpd,'tpp',tpp,'tpp2',tpp2,'ep',ep,'ed',ed, ...
           'epp', ep - 2*tpp*shift);        % effective oxygen level on the diagonal

% default frequency grid: band edges from a coarse k-grid
if isempty(omega)
   [emin, emax] = band_range(P);
   omega = linspace(emin - 2, emax + 2, 801);
end
omega = omega(:);
No = numel(omega);
Ad = zeros(No,1);  Ap = zeros(No,1);

opts = {'AbsTol',AbsTol,'RelTol',RelTol};
parfor it = 1:No
   z = omega(it) + 1i*eta;
   Gd  = integral2(@(kx,ky) Gdd_k(kx,ky,z,P),   0,pi,0,pi, opts{:})/pi^2;
   Gpx = integral2(@(kx,ky) Gpxpx_k(kx,ky,z,P), 0,pi,0,pi, opts{:})/pi^2;
   Ad(it) = -imag(Gd)/pi;
   Ap(it) = -imag(Gpx)/pi;
end
Atot = Ad + 2*Ap;

% ----- report sum rules (should be close to 1, 1, 3; tails of the
%       Lorentzian broadening outside the omega window reduce them slightly) -----
fprintf('sum rules: int A_d = %.4f, int A_p = %.4f, int A_tot = %.4f\n', ...
        trapz(omega,Ad), trapz(omega,Ap), trapz(omega,Atot));

if doplot
   figure; hold on; box on;
   plot(omega, Ad,   'LineWidth', 1.5);
   plot(omega, Ap,   'LineWidth', 1.5);
   plot(omega, Atot, 'k', 'LineWidth', 1.0);
   xlabel('\omega'); ylabel('DOS per spin per orbital');
   legend({'Cu d_{x^2-y^2}', 'O p_x (= p_y)', 'total'}, 'Location', 'best');
   title(sprintf('Emery model: t_{pd}=%g, t_{pp}=%g, t''_{pp}=%g, \\epsilon_p=%g, \\epsilon_d=%g, \\eta=%g', ...
         tpd, tpp, tpp2, ep, ed, eta));
end
end

% =====================================================================
function [b1, b2, D, xx, xy] = oxy_blocks(kx, ky, z, P)
% B = (z - h_pp)^{-1} in the real gauge; returns B*s = (b1,b2), det D and diagonal of h_pp
sx = sin(kx/2);  sy = sin(ky/2);
xx = P.epp + 2*P.tpp2*cos(kx);
xy = P.epp + 2*P.tpp2*cos(ky);
h  = 4*P.tpp.*sx.*sy;
D  = (z - xx).*(z - xy) - h.^2;
b1 = ((z - xy).*sx + h.*sy)./D;
b2 = (h.*sx + (z - xx).*sy)./D;
end

function g = Gdd_k(kx, ky, z, P)
[b1, b2] = oxy_blocks(kx, ky, z, P);
W = 4*P.tpd^2*(sin(kx/2).*b1 + sin(ky/2).*b2);
g = 1./(z - P.ed - W);
end

function gp = Gpxpx_k(kx, ky, z, P)
[b1, b2, D, ~, xy] = oxy_blocks(kx, ky, z, P);
W  = 4*P.tpd^2*(sin(kx/2).*b1 + sin(ky/2).*b2);
g  = 1./(z - P.ed - W);
gp = (z - xy)./D + g.*4*P.tpd^2.*b1.^2;      % B_xx + g * 4 tpd^2 (B s)_x^2
end

function [emin, emax] = band_range(P)
% band edges from exact diagonalization of h0(k) on a coarse grid
k = linspace(-pi, pi, 61);  emin = inf;  emax = -inf;
for kx = k
   for ky = k
      ax = 1 - exp(-1i*kx);  ay = 1 - exp(-1i*ky);
      hxy = P.tpp*(1 - exp(1i*kx))*(1 - exp(-1i*ky));
      H = [P.ed,            P.tpd*ax,                    P.tpd*ay;
           conj(P.tpd*ax),  P.epp + 2*P.tpp2*cos(kx),    hxy;
           conj(P.tpd*ay),  conj(hxy),                   P.epp + 2*P.tpp2*cos(ky)];
      e = eig((H+H')/2);
      emin = min(emin, min(e));  emax = max(emax, max(e));
   end
end
end
