function Wm = Wmix_site(kx,ky,z,P)
% 4x4 cell of arrays: mixed representation of the oxygen-mediated kernel in the
% SITE basis of the 2x2 Cu plaquette,
%   [W(z,k~)]_ab = (1/4) sum_Q exp(i(k~+Q).(r_a - r_b)) W(z, k~+Q),
% sites F1=(0,0), F2=(0,1), F3=(1,0), F4=(1,1); Q in {(0,0),(pi,0),(0,pi),(pi,pi)}.
r  = [0 0; 0 1; 1 0; 1 1];
Qs = [0 0; pi 0; 0 pi; pi pi];
Wm = cell(4,4);
for a = 1:4, for b = 1:4, Wm{a,b} = zeros(size(kx)); end, end
for iQ = 1:4
   qx = kx + Qs(iQ,1);  qy = ky + Qs(iQ,2);
   Wq = emeryW(qx,qy,z,P);
   for a = 1:4
      for b = 1:4
         d = r(a,:) - r(b,:);
         Wm{a,b} = Wm{a,b} + exp(1i*(qx*d(1) + qy*d(2))).*Wq/4;
      end
   end
end
end
