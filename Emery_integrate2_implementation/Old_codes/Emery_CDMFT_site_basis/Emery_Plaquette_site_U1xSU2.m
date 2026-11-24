function [Hepsd,HU,H0,A0,FF,ZF,SF,EF,NF] = Emery_Plaquette_site_U1xSU2(U,mu,ed,varargin)
% < Description >
%
% [Hepsd,HU,H0,A0,FF,ZF,SF,EF,NF] = Emery_Plaquette_site_U1xSU2(U,mu,ed,['-iso2HU'])
%
% Impurity of 2x2-plaquette CDMFT for the Emery model in the SITE basis,
% oxygens integrated out (t_cl = ed*1, no static Cu-Cu hopping):
%
%   F2--F4      F1=(0,0), F2=(0,1), F3=(1,0), F4=(1,1)
%   |    |
%   F1--F3      H = sum_{i,s} (ed-mu) n_is + U sum_i n_i,up n_i,dn
%
% The bath couples to the four site operators FF(1..4) through a
% matrix-valued hybridization Delta_ab(w) (matrix Wilson chain).
%
% < Option >
% '-iso2HU' : contract HU and Hepsd with the isometry A0, giving itags
%             {'K00','K00*'} as required by impSE.

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

NF = quadOp(FF,FF,[]);

% on-site interaction: (U/2) n_i (n_i - 1) = U n_i,up n_i,dn
HU = QSpace;
for i = 1:NC
   HU = HU + (U/2)*(sum(NF(i))*(sum(NF(i)) - EF));   % identity EF, not the number 1
end

% one-body part: (ed - mu) on every site, no static hopping
Hepsd = QSpace;
for i = 1:NC
   Hepsd = Hepsd + (ed - mu)*NF(i);
end

H0 = HU + Hepsd + EF*1e-33;
A0 = getIdentity(setItag('L00',getvac(EF)),2,EF,2,'K00*',[1 3 2]);
H0 = contract(A0,'!2*',{H0,'!1',A0});
if contract_HU
   HU    = contract(A0,'!2*',{HU,'!1',A0});
   Hepsd = contract(A0,'!2*',{Hepsd,'!1',A0});
end
end
