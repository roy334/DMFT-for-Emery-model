function [Hepsd,HU,H0,A0,FF,ZF,SF,EF,NF,FK,NK] = Emery_Plaquette_U1xSU2(U,mu,ed,varargin)
% < Description >
%
% [Hepsd,HU,H0,A0,FF,ZF,SF,EF,NF,FK,NK] = Emery_Plaquette_U1xSU2(U,mu,ed,['-iso2HU'])
%
% Impurity of 2x2-plaquette CDMFT for the Emery model, oxygens integrated
% out (t_cl = ed*1, no static Cu-Cu hopping). Same output order as
% Hubbard_Dimer.m, so it can replace it in the impSE call.
%
%   F2--F4      F1=(0,0), F2=(0,1), F3=(1,0), F4=(1,1)
%   |    |
%   F1--F3      H = sum_{i,s} (ed-mu) n_is + U sum_i n_i,up n_i,dn
%
% The Wilson chains attach to the cluster-momentum operators FK, in which
% the hybridization is diagonal (mirror symmetry of the plaquette):
%   FK(1): K=(0,0)   FK(2): K=(pi,0)   FK(3): K=(0,pi)   FK(4): K=(pi,pi)
% This order must match the columns of Rhoin / SE and WK_matrix.m.
%
% < Option >
% '-iso2HU' : contract HU and Hepsd with the isometry A0 (as in Hubbard_Dimer)

contract_HU = false;
while ~isempty(varargin)
   switch varargin{1}
      case '-iso2HU', contract_HU = true;  varargin(1) = [];
      otherwise, error(['ERR: Unknown input ', num2str(varargin{1})]);
   end
end

NC = 4;
[FF,ZF,SF,IF] = getLocalSpace('FermionS','Acharge,SU2spin','NC',NC);
[FF,ZF,SF,EF] = setItag('s00','op',FF,ZF,SF,IF.E);

% cluster-momentum operators d_K = sum_a U_aK d_a,  U_aK = exp(iK.r_a)/2 (real)
FK = QSpace(1,4);
FK(1) = 0.5*( FF(1) + FF(2) + FF(3) + FF(4));   % (0,0)
FK(2) = 0.5*( FF(1) + FF(2) - FF(3) - FF(4));   % (pi,0)
FK(3) = 0.5*( FF(1) - FF(2) + FF(3) - FF(4));   % (0,pi)
FK(4) = 0.5*( FF(1) - FF(2) - FF(3) + FF(4));   % (pi,pi)

NF = quadOp(FF,FF,[]);
NK = quadOp(FK,FK,[]);

% on-site interaction (diagonal in FF): (U/2) n_i (n_i - 1) = U n_i,up n_i,dn
HU = QSpace;
for i = 1:NC
   HU = HU + (U/2)*(sum(NF(i))*(sum(NF(i)) - EF));   % identity EF
end

% one-body part: t_cl = ed*1 -> (ed - mu) on every channel (diagonal in FK and FF)
Hepsd = QSpace;
for K = 1:NC
   Hepsd = Hepsd + (ed - mu)*NK(K);
end

H0 = HU + Hepsd + EF*1e-33;
A0 = getIdentity(setItag('L00',getvac(EF)),2,EF,2,'K00*',[1 3 2]);
H0 = contract(A0,'!2*',{H0,'!1',A0});
if contract_HU
   HU    = contract(A0,'!2*',{HU,'!1',A0});
   Hepsd = contract(A0,'!2*',{Hepsd,'!1',A0});
end
end
