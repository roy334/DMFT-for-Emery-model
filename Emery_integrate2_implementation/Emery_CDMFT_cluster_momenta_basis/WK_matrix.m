function WK = WK_matrix(kx,ky,z,P,zO)
% 4x4 cell of arrays: W(z,k~) in the cluster-momentum basis of the 2x2 plaquette,
%   [W_K]_{KK'} = (1/16) sum_Q W(z,k~+Q) phi(k~+Q-K) conj(phi(k~+Q-K')),
%   phi(q) = (1+e^{i q_x})(1+e^{i q_y}).
% Channel order K = (0,0), (pi,0), (0,pi), (pi,pi). Independent of how the
% four sites are numbered (phi sums over all sites).
% Optional zO: frequency seen by the oxygens, zO = z - SEeta(w) (default z).
if nargin < 5 || isempty(zO), zO = z; end
Ks  = [0 0; pi 0; 0 pi; pi pi];
phi = @(qx,qy) (1+exp(1i*qx)).*(1+exp(1i*qy));
WK  = cell(4,4);
for a = 1:4, for b = 1:4, WK{a,b} = zeros(size(kx)); end, end
for iQ = 1:4
   qx = kx + Ks(iQ,1);  qy = ky + Ks(iQ,2);
   Wq = emeryW(qx,qy,z,P,zO);
   pK = cell(1,4);
   for a = 1:4, pK{a} = phi(qx-Ks(a,1), qy-Ks(a,2)); end
   for a = 1:4
      for b = 1:4
         WK{a,b} = WK{a,b} + Wq.*pK{a}.*conj(pK{b})/16;
      end
   end
end
end
