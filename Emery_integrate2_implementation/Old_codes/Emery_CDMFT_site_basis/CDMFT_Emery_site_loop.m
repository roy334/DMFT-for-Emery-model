%% CDMFT for the Emery model: 2x2 Cu plaquette, SITE basis
% Oxygens integrated out:  G0eff^-1(z,k) = z - ed - W(z,k),  t_cl = ed*1.
% Matrix-valued hybridization Delta_ab(w) with off-diagonal elements
% -> matrix Wilson chain (doZLD block Lanczos + NRG_SL with 3D ff, gg).
%
% Lattice step (integrateG, site basis, mixed representation):
%   G_ab(w)   = (1/pi^2) int_[0,pi]^2 d^2k~ [ (w-(ed-mu))1 - Sigma - W(w+mu,k~) ]^-1_ab
%   Delta_ab  = w - (ed-mu) - Sigma_ab - [G^-1]_ab
% Sites: F1=(0,0), F2=(0,1), F3=(1,0), F4=(1,1). Fixed mu.


clear
num_threads_SL(8);

%% This hack is required for laptop since I do not have Parallel Computing toolbox

if isempty(ver('parallel'))
   addpath(fullfile(getenv('HOME'),'matlab_noPCT'));
   disp('No Parallel Computing Toolbox: using stand-in gcp, integrateG runs serially.');
end


%% Step 1. Parameters, frequency grid, lattice kernel
P  = struct('ed',0,'ep',8,'tpd',1.5,'tpp',1,'tpp2',1);  % Sordi et al. Eq.(16); diag = ep-2tpp+2tpp2 cos k
U  = 11.1;           % Hubbard U on Cu
mu = 10;             % fixed chemical potential
T  = 1/2;           % temperature
Nc = 4;

% The SAME grid options must be passed to impSE ('getAcont'), because impSE
% regenerates ocont internally and discretizes RhoV2in on it.

optAcont = {'emax',25};                  % must cover ~[-15,10] around mu
ocont    = getAcont(0,0,0,0,'emax',25);

% No static hopping (Ek = 0); oxygen-mediated kernel via 'Fkw' (site basis).
% integrateG passes w = ocont + 1i*iReg
% W is evaluated at z = w + mu.


Ek = cell(4,4);  Fkw = cell(4,4);
for a = 1:4
   for b = 1:4
      Ek{a,b}  = @(kx,ky) 0*kx;
      Fkw{a,b} = @(kx,ky,w) pickElem(Wmix_site(kx,ky,w+mu,P),a,b);
   end
end

% Symmetry classes: 1 on-site, 2 nearest neighbour, 3 diagonal


Pcls = [1 2 2 3; 2 1 3 2; 2 3 1 2; 3 2 2 1];  % This indexing shows which elements are independent. 
                                              % Note because of C4
                                              % symmetry, we have only 3
                                              % indep elements in \Sigma
                                              % and \Delta!

iReg = 1e-4;
intopt = {'Fkw',Fkw,'Hloc',P.ed-mu,'CDMFT',Pcls,'N',1/pi^2,'iReg',iReg,'AbsTol',1e-6,'RelTol',1e-4};

%% Step 2. Starting hybridization from the U = 0 lattice step

fpath = [go('mu'),'/DMFT/Emery_CDMFT'];
fname = @(n) [fpath,'/CDMFT_Emery_site_U',num2str(U),'_mu',num2str(mu),'_T',num2str(T),'_',num2str(n),'.mat'];
N2 = 1;                                  % run counter
if N2 == 1
   [RhoV2,Delta0,~,~] = integrateG(Ek,[0 0],[pi pi],ocont,intopt{:});
   Rhoin = symPlaq(real(RhoV2));         % Noc x 4 x 4, real symmetric (C4v). We enforce C4 symmetry by hand
   RhoV2_arr = [];  nd = [];
else
   load(fname(N2-1),'Rhoin','nd');
   Rhoin = Rhoin(:,:,:,end);  RhoV2_arr = [];
end

%% Step 3. Impurity Hamiltonian and NRG parameters
[Hmu,HU,~,A0,FF,ZF,~,~,NF] = Emery_Plaquette_site_U1xSU2(U,mu,P.ed,'-iso2HU');

Gdep = Pcls;  Fdep = Pcls;               % matrix self-energy, 3 independent classes

useINRG = true;                          % false: exact matrix chain, 256 local states per site
if useINRG
   inrg = 1:Nc;                          % one sub-channel per site; off-diagonal chain terms
else                                     % between sub-channels are geometric-mean approximations
   inrg = [];
end

Lambda  = 7;
nz      = 1;
nrgdata = [getenv('SCRATCH_DIR'),'/Emery_site'];
Nkeep   = 300;                           
NKEEP   = 3*Nkeep*ones(1,3);
Etrunc  = 9;

%% Step 4. DMFT cycle
Nmax  = 40;
iscvg = false;
while (iscvg == false) && (size(Rhoin,4) < Nmax)
   % 4.1 impurity: matrix hybridization -> matrix self-energy SE(:,a,b)
   [ocont_imp,SE,nrgdataz] = impSE(Hmu,HU,A0,FF,ZF,Rhoin(:,:,:,end),T,Lambda,nz,Nkeep,nrgdata, ...
      'inrg',inrg,'Gdep',Gdep,'Fdep',Fdep,'broaden','adap','numCorr',3, ...
      'getAcont',optAcont, ...
      'doZLD',{'emin',1e-33,'minTE',1e-8}, ...
      'NRG_SL',{'Etrunc',Etrunc,'NKEEP',NKEEP}, ...
      'SEtrick',{'iSEmax',-1e-8});
   if numel(ocont_imp) ~= numel(ocont) || any(abs(ocont_imp(:)-ocont(:)) > 1e-12)
      error('ERR: impSE frequency grid differs from the lattice grid; check optAcont.');
   end
   SE = symPlaq(SE);                     % enforce C4 structure
   nd = cat(2,nd,getEpVal(nrgdataz,sum(NF))/Nc);   % Cu filling per site

   % 4.2 lattice step: full 4x4 G_ab and Delta_ab in the site basis
   [RhoV2,Delta,A,G] = integrateG(Ek,[0 0],[pi pi],ocont,intopt{:},'SE',SE);
   RhoV2 = symPlaq(real(RhoV2));                         % enforce C4 structure
   RhoV2_arr = cat(4,RhoV2_arr,RhoV2);

   % 4.3 update matrix hybridization at fixed mu (x = 4: iterations along dim 4)
   [iscvg,Rho_new] = updateHyb(Rhoin,RhoV2_arr,4,'Broyden',2,'amix',0.9,'ocont',ocont,'-v');
   if ~iscvg
      Rhoin = cat(4,Rhoin,Rho_new);
   end

   save(fname(N2),'-v7.3');
end
