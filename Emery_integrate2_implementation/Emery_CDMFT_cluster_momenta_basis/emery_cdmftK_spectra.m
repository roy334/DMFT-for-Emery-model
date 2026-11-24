function [nd, np, ntot, AK, Ap, GK] = emery_cdmftK_spectra(w, mu, P, SEK, T, varargin)
% < Description >
%
% [nd, np, ntot, AK, Ap, GK] = emery_cdmftK_spectra(w, mu, P, SEK, T [,options])
%
% Spectra and occupations of the Emery model in 2x2-plaquette CDMFT (oxygens
% integrated out, U_p = 0) from the cluster self-energy SEK in the K basis.
% k~-integration over one reduced zone [0,pi]^2, z = w + i*eta + mu, either on a fixed
% midpoint grid ('Method','grid', default) or adaptively ('Method','adaptive': nested
% 1D Gauss-Kronrod (7,15) in k~y and k~x, as in latticeStepK_asym.m). The frequency
% loop is a parfor (uses the open parallel pool; none open = serial).
%
%   Cu:  G(k~) = [ (z-ed) 1 - W_K(z,k~) - diag(SEK) ]^-1,   A_K = -Im <G_KK>/pi
%   O:   with A_Q = (zO - h_pp(k~+Q))^{-1} (2x2), zO = z - SEeta(w), and
%        u_Q(K) = phi(k~+Q-K)/4 the plane-wave vectors of WK_matrix.m
%        (W_K = sum_Q W(k~+Q) u_Q u_Q^dag),
%        sum over all 8 O orbitals of the supercell:
%        Tr G_OO(k~) = sum_Q [ tr A_Q + (u_Q^dag G(k~) u_Q) * w A_Q^2 w^dag ]
%        A_p = -Im <Tr G_OO>/(8 pi)       (per O orbital, averaged over all oxygens,
%                                           intra- and inter-plaquette alike)
%   nd = 2 * (1/4) sum_K int f A_K  (per Cu),   np = 2 int f A_p,   ntot = nd + 2 np
%
% < Option >
% 'eta', ..   : broadening (default 0.02; the bare O poles need a finite eta)
% 'Method', ..: 'grid' (default) or 'adaptive'
% 'Nk', ..    : grid only: k~ points per direction (default 128)
% 'RelTol', ..: adaptive only: relative tolerance of the outer (k~x) integral
%               (default 1e-4); the inner (k~y) integrals use RelTol/10
% 'AbsTol', ..: adaptive only: absolute tolerance (default 1e-8)
% 'MaxIntervals', .. : adaptive only: max. subintervals per 1D integral (default 650)
% 'SEeta', .. : O broadening on w (vector). Default [] = removed (physical U_p = 0
%               oxygens); pass the loop's SEeta(w) to reproduce the loop's lattice.
% '-v'        : print the total time
%
% < Output >
% nd, np, ntot ; AK : numel(w) x 4 (Cu, K channels) ; Ap : numel(w) x 1 ;
% GK : numel(w) x 4 local cluster Green's function

eta = 0.02;  Nk = 128;  SEeta = [];  verbose = false;
Method = 'grid';  RelTol = 1e-4;  AbsTol = 1e-8;  MaxIntervals = 650;
while ~isempty(varargin)
   switch varargin{1}
      case 'Method', Method = varargin{2}; varargin(1:2) = [];
      case 'RelTol', RelTol = varargin{2}; varargin(1:2) = [];
      case 'AbsTol', AbsTol = varargin{2}; varargin(1:2) = [];
      case 'MaxIntervals', MaxIntervals = varargin{2}; varargin(1:2) = [];
      case 'eta',   eta   = varargin{2}; varargin(1:2) = [];
      case 'Nk',    Nk    = varargin{2}; varargin(1:2) = [];
      case 'SEeta', SEeta = varargin{2}; varargin(1:2) = [];
      case '-v',    verbose = true;      varargin(1)   = [];
      otherwise, error(['ERR: unknown option ',num2str(varargin{1})]);
   end
end
w = w(:);  Nw = numel(w);
if size(SEK,2) == 3, SEK = SEK(:,[1 2 2 3]); end
if isempty(SEeta), SEeta = zeros(Nw,1); end
SEeta = SEeta(:);

GK = zeros(Nw,4);  TrO = zeros(Nw,1);
t0 = tic;

if strcmpi(Method,'adaptive')
   nfail = 0;
   parfor it = 1:Nw                              % frequencies are independent
      z  = w(it) + 1i*eta + mu;
      zO = z - SEeta(it);
      zd = z - P.ed - SEK(it,:);
      fpt   = @(kx,ky) kpoint_spec(kx,ky,z,zO,zd,P);
      inner = @(kxv,~) inner_int(fpt,kxv,RelTol/10,AbsTol/10,MaxIntervals);
      [Iv,~,nf] = gk_family(inner,0,pi,1,RelTol,AbsTol,MaxIntervals,10);
      Iv = Iv/pi^2;
      GK(it,:) = Iv(1:4).';
      TrO(it)  = Iv(5);
      nfail = nfail + nf;
   end
   if nfail > 0
      fprintf('WRN: emery_cdmftK_spectra: %d outer k~x integrals hit MaxIntervals.\n', nfail);
   end
else
k = ((1:Nk) - 0.5)*pi/Nk;
[KX,KY] = meshgrid(k,k);  KX = KX(:).';  KY = KY(:).';
NK = numel(KX);
Ks  = [0 0; pi 0; 0 pi; pi pi];
phi = @(qx,qy) (1+exp(1i*qx)).*(1+exp(1i*qy));

% plane-wave vectors u_Q(K) (independent of frequency)
u = zeros(4,1,NK,4);                              % (K, 1, k~, Q)
for iQ = 1:4
   qx = KX + Ks(iQ,1);  qy = KY + Ks(iQ,2);
   for a = 1:4
      u(a,1,:,iQ) = reshape(phi(qx-Ks(a,1), qy-Ks(a,2))/4, 1,1,NK);
   end
end

parfor it = 1:Nw                                 % frequencies are independent
   z  = w(it) + 1i*eta + mu;
   zO = z - SEeta(it);
   W  = zeros(4,4,NK);  trA = zeros(1,NK);  wA2 = zeros(4,NK);
   for iQ = 1:4
      qx = KX + Ks(iQ,1);  qy = KY + Ks(iQ,2);
      [Wq, wA2w, trAinv] = emeryW(qx,qy,z,P,zO);
      uQ = u(:,:,:,iQ);
      W  = W + reshape(Wq,1,1,NK).*pagemtimes(uQ, pagectranspose(uQ));
      trA = trA + trAinv;
      wA2(iQ,:) = wA2w;
   end
   M = -W;
   for a = 1:4
      M(a,a,:) = M(a,a,:) + (z - P.ed - SEK(it,a));
   end
   Gk = pageinv(M);
   for a = 1:4
      GK(it,a) = mean(Gk(a,a,:));                % diagonal elements only
   end
   t = trA;
   for iQ = 1:4
      uQ  = u(:,:,:,iQ);
      uGu = reshape(sum(conj(uQ).*pagemtimes(Gk,uQ),1), 1, NK);   % u_Q^dag G u_Q
      t = t + uGu.*wA2(iQ,:);
   end
   TrO(it) = mean(t);
end
end
if verbose
   fprintf('emery_cdmftK_spectra: %d frequencies, %.1f s\n', Nw, toc(t0));
end

AK = -imag(GK)/pi;
Ap = -imag(TrO)/(8*pi);
fw = 1./(exp(w/T)+1);
nd = 2*mean(trapz(w, fw.*AK));
np = 2*trapz(w, fw.*Ap);
ntot = nd + 2*np;
end


function V = kpoint_spec(kx,ky,z,zO,zd,P)
% integrand at the k~ points (kx,ky) (row vectors): rows 1-4 G_KK(k~),
% row 5 Tr G_OO(k~) = sum_Q [ tr A_Q + (u_Q^dag G u_Q) * w A_Q^2 w^dag ]
NK  = numel(kx);
Ks  = [0 0; pi 0; 0 pi; pi pi];
phi = @(qx,qy) (1+exp(1i*qx)).*(1+exp(1i*qy));
u   = zeros(4,NK,4);                              % (K, k~, Q)
W   = zeros(4,4,NK);  t = zeros(1,NK);  wA2 = zeros(4,NK);
for iQ = 1:4
   qx = kx + Ks(iQ,1);  qy = ky + Ks(iQ,2);
   for a = 1:4, u(a,:,iQ) = phi(qx-Ks(a,1), qy-Ks(a,2))/4; end
   [Wq, wA2w, trAinv] = emeryW(qx,qy,z,P,zO);
   for a = 1:4
      for b = 1:4
         W(a,b,:) = W(a,b,:) + reshape(Wq.*u(a,:,iQ).*conj(u(b,:,iQ)),1,1,NK);
      end
   end
   t = t + trAinv;
   wA2(iQ,:) = wA2w;
end
M = -W;
for a = 1:4
   M(a,a,:) = M(a,a,:) + zd(a);
end
if exist('pageinv') > 0
   Gk = pageinv(M);
else
   Gk = zeros(size(M));
   for ik = 1:NK, Gk(:,:,ik) = inv(M(:,:,ik)); end
end
Gm = reshape(Gk,16,NK);                           % (a,b) -> a + 4(b-1)
for iQ = 1:4
   uGu = zeros(1,NK);                             % u_Q^dag G u_Q
   for a = 1:4
      for b = 1:4
         uGu = uGu + conj(u(a,:,iQ)).*Gm(a+4*(b-1),:).*u(b,:,iQ);
      end
   end
   t = t + uGu.*wA2(iQ,:);
end
V = [Gm([1 6 11 16],:); t];
end

function V = inner_int(fpt,kxv,RelTol,AbsTol,MaxIntervals)
% k~y integrals over [0,pi] for every k~x in kxv, batched: m x numel(kxv)
[V,~,nf] = gk_family(@(ky,id) fpt(kxv(id),ky),0,pi,numel(kxv),RelTol,AbsTol,MaxIntervals,10);
if nf > 0
   warning('emery_cdmftK_spectra:inner','%d inner k~y integrals hit MaxIntervals.',nf);
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

