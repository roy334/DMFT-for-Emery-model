function [RhoV2,Delta,G,F,info] = latticeStepK_asym(ocont,mu,P,SEK,varargin)
% < Description >
%
% [RhoV2,Delta,G,F,info] = latticeStepK_asym(ocont,mu,P,SEK [,options])
%
% CDMFT lattice step for the Emery 2x2 Cu plaquette with the oxygens
% integrated out, in the cluster-momentum (K) basis, with the ASYMMETRIC
% ESTIMATOR for the hybridization and an optional causal O broadening SEeta.
% k~-integration over one reduced zone [0,pi]^2, batched 4x4 inversions, either on a
% fixed midpoint grid ('Method','grid', default) or adaptively ('Method','adaptive'):
% nested 1D adaptive Gauss-Kronrod (7,15) in k~y (inner) and k~x (outer), all inner
% integrals of one refinement round batched into a single pageinv call. The adaptive
% rule puts points where the integrand is sharp (the Fermi-surface ridge of width
% ~ eta + |Im Sigma|), so it stays accurate for small eta in a metal.
%
%   W_K(z,k~)  = oxygen kernel in the K basis, oxygens at zO = z - SEeta(w)
%   G(k~,z)    = [ (z - ed) 1 - W_K(z,k~) - diag(SEK) ]^-1     (full 4x4 at each k~)
%   G_K        = < G_KK(k~,z) >_k~
%   F_K        = < [W_K G]_KK >_k~ = < sum_K' [W_K]_KK' G_K'K >_k~   (diagonal only)
%   Delta_K    = F_K / G_K ,   RhoV2_K = max(-Im Delta_K/pi, 0),   z = w + i*eta + mu
%
% Impurity level ed for every channel (no static Cu-Cu hopping), so nothing is
% subtracted. Exact identity: F_K = (z - ed - SEK) G_K - 1 (info.idErr checks it).
% <G>_k~ is diagonal in K by the plaquette's mirror symmetries; with 'checkOffK'
% the off-diagonal elements are averaged too, as a check (info.offK).
%
% < Input >
% ocont : frequency grid (measured from mu)
% mu    : chemical potential
% P     : struct with ed, ep, tpd, tpp, tpp2 (as for emeryW.m)
% SEK   : Noc x 4 cluster self-energy, K = (0,0),(pi,0),(0,pi),(pi,pi)
%         (Noc x 3 = [(0,0), (pi,0)=(0,pi), (pi,pi)] also accepted; [] = 0)
%
% < Option >
% 'Method', ..: 'grid' (default) or 'adaptive'
% 'Nk', ..    : grid only: k~ points per direction (default 128). Check convergence vs eta.
% 'RelTol', ..: adaptive only: relative tolerance of the outer (k~x) integral
%               (default 1e-4); the inner (k~y) integrals use RelTol/10
% 'AbsTol', ..: adaptive only: absolute tolerance (default 1e-8)
% 'MaxIntervals', .. : adaptive only: max. subintervals per 1D integral (default 650)
% 'eta', ..   : constant broadening of z (default 0.02); a fixed grid needs
%               eta + |Im Sigma| >~ (band velocity) x pi/Nk, the adaptive rule does not
% 'SEeta', .. : causal O broadening on ocont (vector, numel(ocont)), inside W only;
%               [] (default) = none
% 'checkOffK' : also average the off-diagonal elements of G and report the largest
%               (info.offK). Off by default: only the 4 diagonal elements of G and F
%               are averaged.
% '-v'        : print the total time of the lattice step
%
% The frequency loop is a parfor: it uses the open parallel pool (none open = serial).
%
% < Output >
% RhoV2, Delta, G, F : Noc x 4 (columns K); (pi,0) and (0,pi) averaged in RhoV2, Delta
% info.idErr, info.offK (NaN unless 'checkOffK'), info.time,
% adaptive only: info.neval (k~ points per frequency), info.nfail (number of 1D
% integrals that hit MaxIntervals, summed over frequencies; should be 0)
%
% Uses WK_matrix.m and emeryW.m (both with the optional zO argument).

Nk = 128;  eta = 0.02;  SEeta = [];  verbose = false;  checkOffK = false;
Method = 'grid';  RelTol = 1e-4;  AbsTol = 1e-8;  MaxIntervals = 650;
while ~isempty(varargin)
   switch varargin{1}
      case 'Method', Method = varargin{2}; varargin(1:2) = [];
      case 'RelTol', RelTol = varargin{2}; varargin(1:2) = [];
      case 'AbsTol', AbsTol = varargin{2}; varargin(1:2) = [];
      case 'MaxIntervals', MaxIntervals = varargin{2}; varargin(1:2) = [];
      case 'Nk',    Nk    = varargin{2}; varargin(1:2) = [];
      case 'eta',   eta   = varargin{2}; varargin(1:2) = [];
      case 'SEeta', SEeta = varargin{2}; varargin(1:2) = [];
      case '-v',    verbose = true;      varargin(1)   = [];
      case 'checkOffK', checkOffK = true; varargin(1)  = [];
      otherwise, error(['ERR: unknown option ',num2str(varargin{1})]);
   end
end

ocont = ocont(:);  Noc = numel(ocont);
if isempty(SEK), SEK = zeros(Noc,4); end
if size(SEK,2) == 3, SEK = SEK(:,[1 2 2 3]); end
if ~isequal(size(SEK),[Noc 4]), error('ERR: SEK must be Noc x 4 (or Noc x 3).'); end
if isempty(SEeta), SEeta = zeros(Noc,1); end
SEeta = SEeta(:);
if numel(SEeta) ~= Noc, error('ERR: ''SEeta'' must have numel(ocont) elements.'); end

usePage = exist('pageinv') > 0;                  % MATLAB R2022a+

G = zeros(Noc,4);  F = G;  offK = NaN(Noc,1);
t0 = tic;

if strcmpi(Method,'adaptive')
   neval = zeros(Noc,1);  nfail = 0;
   parfor it = 1:Noc                             % frequencies are independent
      z  = ocont(it) + 1i*eta + mu;
      zO = z - SEeta(it);
      zd = z - P.ed - SEK(it,:);
      fpt = @(kx,ky) kpoint_vals(kx,ky,z,zO,zd,P,usePage,checkOffK);
      % inner: k~y integrals for all outer nodes k~x at once; outer: k~x
      inner = @(kxv,~) inner_int(fpt,kxv,RelTol/10,AbsTol/10,MaxIntervals);
      kpoint_vals('reset');
      [Iv,~,nf] = gk_family(inner,0,pi,1,RelTol,AbsTol,MaxIntervals,10);
      neval(it) = kpoint_vals('count');
      Iv = Iv/pi^2;
      G(it,:) = Iv(1:4).';
      F(it,:) = Iv(5:8).';
      if checkOffK, offK(it) = max(abs(Iv(9:end))); end
      nfail = nfail + nf;
   end
   info.neval = neval;  info.nfail = nfail;
   if verbose
      fprintf('latticeStepK_asym (adaptive): %d frequencies, %.1f s\n', Noc, toc(t0));
   end
else
k = ((1:Nk) - 0.5)*pi/Nk;                        % midpoints of [0,pi] (one reduced zone)
[KX,KY] = meshgrid(k,k);  KX = KX(:).';  KY = KY(:).';
NK = numel(KX);
parfor it = 1:Noc                                % frequencies are independent
   z  = ocont(it) + 1i*eta + mu;
   zO = z - SEeta(it);
   Wc = WK_matrix(KX,KY,z,P,zO);                 % 4x4 cell of 1 x NK arrays
   W  = zeros(4,4,NK);
   for a = 1:4
      for b = 1:4
         W(a,b,:) = reshape(Wc{a,b},1,1,NK);
      end
   end
   M = -W;
   for a = 1:4
      M(a,a,:) = M(a,a,:) + (z - P.ed - SEK(it,a));
   end
   if usePage
      Gk = pageinv(M);
      Gt = pagetranspose(Gk);
   else
      Gk = zeros(size(M));  Gt = Gk;
      for ik = 1:NK, Gk(:,:,ik) = inv(M(:,:,ik)); Gt(:,:,ik) = Gk(:,:,ik).'; end
   end
   Fd = sum(W.*Gt, 2);                           % [W G]_KK = sum_K' W_KK' G_K'K  (4 x 1 x NK)
   for a = 1:4
      G(it,a) = mean(Gk(a,a,:));                 % diagonal elements only
   end
   F(it,:) = mean(Fd,3).';
   if checkOffK
      Gl = mean(Gk,3);
      offK(it) = max(max(abs(Gl - diag(diag(Gl)))));
   end
end
if verbose
   fprintf('latticeStepK_asym: %d frequencies, %.1f s\n', Noc, toc(t0));
end
end

Delta = F./G;
Delta(:,[2 3]) = repmat(mean(Delta(:,[2 3]),2),1,2);          % (pi,0) = (0,pi)
RhoV2 = max(-imag(Delta)/pi, 0);

zeta = ocont + 1i*eta + mu - P.ed - SEK;
info.idErr = max(abs(F - (zeta.*G - 1)),[],1) ./ max(abs(F),[],1);
if checkOffK, info.offK = max(offK) / max(abs(G(:))); else, info.offK = NaN; end
info.time  = toc(t0);
end


function V = kpoint_vals(kx,ky,z,zO,zd,P,usePage,checkOffK)
% integrand at the k~ points (kx,ky) (row vectors): rows 1-4 G_KK, rows 5-8 [W G]_KK,
% rows 9-20 (only with checkOffK) the off-diagonal G_KK'
% kpoint_vals('reset') / kpoint_vals('count'): counter of evaluated k~ points
persistent cnt
if isempty(cnt), cnt = 0; end
if ischar(kx)
   if strcmp(kx,'reset'), cnt = 0; end
   V = cnt;  return
end
NK = numel(kx);  cnt = cnt + NK;
Wc = WK_matrix(kx,ky,z,P,zO);
W  = zeros(4,4,NK);
for a = 1:4
   for b = 1:4
      W(a,b,:) = reshape(Wc{a,b},1,1,NK);
   end
end
M = -W;
for a = 1:4
   M(a,a,:) = M(a,a,:) + zd(a);
end
if usePage
   Gk = pageinv(M);
else
   Gk = zeros(size(M));
   for ik = 1:NK, Gk(:,:,ik) = inv(M(:,:,ik)); end
end
Gm = reshape(Gk,16,NK);                       % column-major: (a,b) -> a + 4(b-1)
Wm = reshape(W,16,NK);
dg = [1 6 11 16];
Fd = zeros(4,NK);
for a = 1:4
   for b = 1:4
      Fd(a,:) = Fd(a,:) + Wm(a+4*(b-1),:).*Gm(b+4*(a-1),:);   % sum_b W_ab G_ba
   end
end
V = [Gm(dg,:); Fd];
if checkOffK
   V = [V; Gm(setdiff(1:16,dg),:)];
end
end

function V = inner_int(fpt,kxv,RelTol,AbsTol,MaxIntervals)
% k~y integrals over [0,pi] for every k~x in kxv, batched: m x numel(kxv)
[V,~,nf] = gk_family(@(ky,id) fpt(kxv(id),ky),0,pi,numel(kxv),RelTol,AbsTol,MaxIntervals,10);
if nf > 0
   warning('latticeStepK_asym:inner','%d inner k~y integrals hit MaxIntervals.',nf);
end
end

function [I,E,nfail,neval] = gk_family(fun,a,b,nfam,RelTol,AbsTol,MaxIntervals,N0)
% Batched adaptive Gauss-Kronrod (7,15) for nfam vector-valued integrals over [a,b].
% fun(x,id): x, id row vectors of equal length (id = which integral) -> m x numel(x).
% Each integral starts from N0 equal subintervals; a subinterval is accepted when its
% error |K15 - G7| (max over components) is below tol*(length/(b-a)), with
% tol = max(AbsTol, RelTol*max|I|) of its own integral, or when the total error of its
% integral is below tol. All active subintervals of all integrals are evaluated in one
% call of fun per round. An integral that exceeds MaxIntervals is accepted as is and
% counted in nfail.
xk = [-0.991455371120813 -0.949107912342759 -0.864864423359769 -0.741531185599394 ...
      -0.586087235467691 -0.405845151377397 -0.207784955007898  0 ...
       0.207784955007898  0.405845151377397  0.586087235467691  0.741531185599394 ...
       0.864864423359769  0.949107912342759  0.991455371120813].';
wk = [ 0.022935322010529  0.063092092629979  0.104790010322250  0.140653259715525 ...
       0.169004726639267  0.190350578064785  0.204432940075298  0.209482141084728 ...
       0.204432940075298  0.190350578064785  0.169004726639267  0.140653259715525 ...
       0.104790010322250  0.063092092629979  0.022935322010529];
wg = [0 0.129484966168870 0 0.279705391489277 0 0.381830050505119 0 0.417959183673469 ...
      0 0.381830050505119 0 0.279705391489277 0 0.129484966168870 0];

e0 = linspace(a,b,N0+1);
L  = repmat(e0(1:end-1),1,nfam);
R  = repmat(e0(2:end),1,nfam);
id = reshape(repmat(1:nfam,N0,1),1,[]);
I = [];  E = zeros(1,nfam);  nint = N0*ones(1,nfam);  nfail = 0;  neval = 0;
while ~isempty(L)
   n  = numel(L);
   c  = (L+R)/2;  h = (R-L)/2;
   x  = c + xk*h;                                 % 15 x n
   ix = repmat(id,15,1);
   f  = fun(x(:).',ix(:).');                      % m x 15n
   neval = neval + 15*n;
   m  = size(f,1);
   f  = reshape(f,m*15,n);
   K  = reshape(kron(wk,eye(m))*f,m,n).*h;        % m x n
   Gs = reshape(kron(wg,eye(m))*f,m,n).*h;
   e  = max(abs(K-Gs),[],1);                      % 1 x n
   if isempty(I), I = zeros(m,nfam); end
   S  = sparse(1:n,id,1,n,nfam);
   Itot = I + K*S;
   Etot = E + e*S;
   tol  = max(AbsTol, RelTol*max(abs(Itot),[],1));        % 1 x nfam
   done = Etot <= tol;
   over = ~done & (nint + full(sum(S,1)) > MaxIntervals);  % would exceed the cap
   nfail = nfail + nnz(over);
   ok = done(id) | over(id) | (e <= tol(id).*(2*h/(b-a)));
   I  = I + K(:,ok)*S(ok,:);
   E  = E + e(ok)*S(ok,:);
   nb = ~ok;                                      % bisect the rest
   nint = nint + full(sum(S(nb,:),1));
   L  = [L(nb) c(nb)];
   R  = [c(nb) R(nb)];
   id = [id(nb) id(nb)];
end
end

