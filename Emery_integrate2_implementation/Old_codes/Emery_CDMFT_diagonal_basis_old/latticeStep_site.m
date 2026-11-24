function [RhoV2,Delta,A,G] = latticeStep_site(ocont,mu,P,SE,varargin)
% < Description >
%
% [RhoV2,Delta,A,G] = latticeStep_site(ocont,mu,P,SE [,options])
%
% Fast CDMFT lattice step for the Emery 2x2 plaquette in the SITE basis,
% using a fixed k~-grid (midpoint rule over one reduced Brillouin zone,
% [0,pi]^2) and batched 4x4 inversions. Same outputs as
%   integrateG(Ek,[0 0],[pi pi],ocont,'Fkw',Fkw,'Hloc',ed-mu,'CDMFT','full',...)
%
%   G_ab(w)   = < [ (z - ed) 1 - W_mix(z,k~) - Sigma(w) ]^-1 >_k~ ,  z = w + i*eta + mu
%   Delta_ab  = (z - ed) 1 - Sigma - G^-1
%   A         = (G - G')/(-2i*pi),   RhoV2 = (Delta - Delta')/(-2i*pi)  (PSD-projected)
%
% < Input >
% ocont : frequency grid (measured from mu)
% mu    : chemical potential
% P     : struct with fields ed, ep, tpd, tpp, tpp2 (as for emeryW.m)
% SE    : Noc x 4 x 4 self-energy, or [] for Sigma = 0
%
% < Option >
% 'Nk', ..  : k~ points per direction (default 64; check convergence vs eta)
% 'eta', .. : broadening (default 0.02). Needs eta >~ few * (bandwidth/Nk).
% '-v'      : print progress
%
% Sites: F1=(0,0), F2=(0,1), F3=(1,0), F4=(1,1)  (see Wmix_site.m)

Nk = 64;  eta = 0.02;  verbose = false;
while ~isempty(varargin)
   switch varargin{1}
      case 'Nk',  Nk  = varargin{2}; varargin(1:2) = [];
      case 'eta', eta = varargin{2}; varargin(1:2) = [];
      case '-v',  verbose = true;    varargin(1)   = [];
      otherwise, error(['ERR: unknown option ',num2str(varargin{1})]);
   end
end

ocont = ocont(:);  Noc = numel(ocont);
if isempty(SE), SE = zeros(Noc,4,4); end
k = ((1:Nk) - 0.5)*pi/Nk;                       % midpoints of [0,pi]
[KX,KY] = meshgrid(k,k);  KX = KX(:).';  KY = KY(:).';   % 1 x Nk^2
NK = numel(KX);
usePage = exist('pageinv') > 0;                 % MATLAB R2022a+

G = zeros(Noc,4,4);  Delta = G;  A = G;  RhoV2 = G;
I4 = eye(4);
t0 = tic;
for it = 1:Noc
   z  = ocont(it) + 1i*eta + mu;
   Wm = Wmix_site(KX,KY,z,P);                   % 4x4 cell of 1 x NK arrays
   S  = reshape(SE(it,:,:),4,4);
   M  = zeros(4,4,NK);
   for a = 1:4
      for b = 1:4
         M(a,b,:) = reshape(-Wm{a,b} - S(a,b) + (a==b)*(z - P.ed), 1,1,NK);
      end
   end
   if usePage
      Gk = pageinv(M);
   else
      Gk = zeros(size(M));
      for ik = 1:NK, Gk(:,:,ik) = inv(M(:,:,ik)); end
   end
   Gl = mean(Gk,3);
   Dl = (z - P.ed)*I4 - S - inv(Gl);
   Al = (Gl - Gl')/(-2i*pi);
   Rl = (Dl - Dl')/(-2i*pi);
   Rl = (Rl + Rl')/2;
   [V,E] = eig(Rl);  E = max(real(diag(E)),0);    % positive semi-definite projection
   Rl = V*diag(E)*V';
   G(it,:,:) = Gl;  Delta(it,:,:) = Dl;  A(it,:,:) = Al;  RhoV2(it,:,:) = Rl;
   if verbose && (mod(it,200)==0 || it==Noc)
      fprintf('latticeStep_site: %d/%d frequencies, %.1f s\n', it, Noc, toc(t0));
   end
end
end
