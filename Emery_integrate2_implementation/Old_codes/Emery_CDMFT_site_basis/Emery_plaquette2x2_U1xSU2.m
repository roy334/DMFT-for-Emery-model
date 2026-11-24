function Himp = Emery_plaquette2x2_U1xSU2(U,mu,ed)
% < Description >
%
% Himp = Emery_plaquette2x2_U1xSU2(U,mu,ed)
%
% Hubbard 2x2 plaquette given by:   F2--F4
%                                   |    |
%                                   |    |
%                                   F1--F3
%
%   H = Sum_{i} (ed-mu)*(n_{i,s}) +Sum_{i} U*(n_{i,up} n_{i,down})
%
% < Input >
% U : [double] Hubbard interaction
% mu : [double] chemical potential
% EK : [double] Onsite potential for the Cu atoms
%
% < Output >
% Himp : [struct] impurity hamiltonian 
%
% Written by S.Roy 

% test values
%U = 1;
%mu = 1.2;

Himp = struct;
Himp.U = U;
Himp.mu = mu;
Himp.ed = ed;
  

pt = 'FermionS';
sym = 'Acharge,SU2spin';
NC = 4;  %Size of the cluster
Himp.locSpace = {pt, sym, NC};

[FF,ZF,SF,IF] = getLocalSpace(pt,sym,'NC',NC);
[FF,ZF,SF,EF] = setItag('s00','op',FF,ZF,SF,IF.E);

NF = quadOp(FF,FF,[]);

% on-site Hubbard interaction: (U/2) n_i (n_i - 1) = U n_i,up n_i,dn

HU = QSpace;
for ito = 1:NC
   HU = HU + (U/2)*(sum(NF(ito))*(sum(NF(ito)) - EF));   
end

% one-body part: (ed - mu) * N, identical on all four sites
Hepsd = (ed - mu)*sum(NF);

H0 = HU + Hepsd + EF*1e-33;
A0 = getIdentity(setItag('L00',getvac(EF)),2,EF,2,'K00*',[1 3 2]);
H0 = contract(A0,'!2*',{H0,'!1',A0});

Himp.HU = HU;   Himp.Hepsd = Hepsd;   Himp.H0 = H0;   Himp.A0 = A0;
Himp.FF = FF;   Himp.NF = NF;   Himp.ZF = ZF;   Himp.EF = EF;   Himp.SF = SF;   Himp.IF = IF;
end

