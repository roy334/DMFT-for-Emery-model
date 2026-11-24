%% CDMFT for the Emery model: 2x2 Cu plaquette, CLUSTER-MOMENTUM (K) basis

% Oxygens integrated out:  G0eff^-1(z,k) = z - ed - W(z,k),  t_cl = ed*1.
% In the K basis  d_K = sum_a U_aK d_a,  U = 1/2*[1 1 1 1; 1 1 -1 -1; 1 -1 1 -1; 1 -1 -1 1],
% the hybridization Delta_K(w) is diagonal -> four independent Wilson chains.
%   columns / channels: K = (0,0), (pi,0), (0,pi), (pi,pi)
%   sites:              F1=(0,0), F2=(0,1), F3=(1,0), F4=(1,1)
%
% Lattice step, directly in the K basis (latticeStepK_asym.m, adaptive k~-integration,
% parfor over frequencies):
%   W_K(z,k~)  = U*W_mix(z,k~)*U            (WK_matrix.m; NOT diagonal at fixed k~)
%   G(k~,z)    = [ (z-ed)1 - W_K(z,k~) - diag(Sigma_K) ]^-1
%   G_K        = < G_KK >_k~ ,   F_K = < [W_K G]_KK >_k~
%   Delta_K    = F_K / G_K        (asymmetric estimator; impurity level ed, nothing subtracted)
% Oxygen broadening: the oxygens inside W see zO = z - SEeta(w), with the causal
%   two-pole SEeta (Kramers-Kronig exact); removable after convergence (removeSEeta).
% Convergence: normalized Wasserstein-1 distance per K channel.
% Fixed mu.


clear
% cores: from the scheduler on a cluster (SLURM_CPUS_PER_TASK), otherwise set by hand
% for the local machine. NRG (client) and the lattice step (pool) run one after the
% other, so both may use all cores.
ncores = str2double(getenv('SLURM_CPUS_PER_TASK'));
if isnan(ncores), ncores = 8; end         % local machine (MacBook Air M4: 8 of 10 cores)
nthr   = 2;                               % threads per pool worker
nwork  = max(1, floor(ncores/nthr));      % workers x threads = ncores
num_threads_SL(ncores);

% parallel pool for the parfor over frequencies in the lattice step and spectra
pool = gcp('nocreate');
if isempty(pool)
   clus = parcluster('Processes');
   clus.NumThreads = nthr;
   clus.JobStorageLocation = tempname;    % unique per job: no clashes between jobs
   mkdir(clus.JobStorageLocation);
   pool = parpool(clus, nwork);
end
fprintf('cores = %d: NRG with %d threads, pool with %d workers x %d threads\n', ...
        ncores, ncores, pool.NumWorkers, nthr);

%% Step 1. Parameters, frequency grid, lattice kernel in the K basis
P  = struct('ed',0,'ep',8,'tpd',1.5,'tpp',1,'tpp2',1);  % Sordi et al. Eq.(16); diag = ep-2tpp+2tpp2 cos k
U  = 11.1;           % Hubbard U on Cu
mu = 10;             % fixed chemical potential
T  = 0.05;           % temperature
Nc = 4;

% Pass SAME grid options to impSE ('getAcont')
optAcont = {'emax',25};                  % must cover ~[-15,10] around mu, estimate from the bare DOS plots
ocont    = getAcont(0,0,0,0,optAcont{:});
Noc      = numel(ocont);


% Causal O broadening inside W (U_p = 0 oxygens have no self-energy of their own):
%   SEeta(z) = etamx*wb*z/((z + i wc)(z + i wb)),  poles at -i wc, -i wb (causal, KK exact)
%   -Im SEeta: 0 at w = 0, max etamx*wb/(wb+wc) = 0.24 at |w| = sqrt(wc*wb) = 10, ~1/w^2 beyond wb
etamx  = 0.3;  wc = 5;  wb = 20;
SEeta  = @(w) etamx*wb*w(:)./((w(:)+1i*wc).*(w(:)+1i*wb));

removeSEeta = true;   % after convergence: spectra/occupations without SEeta (physical oxygens)

% lattice step in the K basis over one reduced zone [0,pi]^2 with adaptive
% k~-integration (nested Gauss-Kronrod), W evaluated at z = w + i*eta + mu, oxygens at
% zO = z - SEeta(w). The adaptive rule resolves the Fermi-surface ridge for small eta;
% {'Method','grid','Nk',128} gives the previous fixed grid.
latopt = {'Method','adaptive','RelTol',1e-4,'eta',0.02,'SEeta',SEeta(ocont)};   % add 'checkOffK' to test K-diagonality

% convergence: normalized Wasserstein-1 distance between input and output
% hybridization of every K channel, all below tolW for Ncons consecutive iterations
tolW  = 5e-3;
Ncons = 3;
W1fun = @(w,a,b) trapz(w, abs(cumtrapz(w,a)/trapz(w,a) - cumtrapz(w,b)/trapz(w,b)));

%% Step 2. Starting hybridization (K basis, Noc x 4)
% startHyb = 'metallic' : non-interacting (Sigma = 0) hybridization, evaluated at a
%                         reference chemical potential mu0 (mu0 = mu: U = 0 bath at the
%                         actual mu; mu0 inside the Cu-like band: generic metallic bath,
%                         as in the DCA example scripts)
% startHyb = 'hubbardI' : Hubbard-I self-energy with n_d determined self-consistently
%                         at this mu (targets the insulating / pseudogap branch)
startHyb = 'metallic';
mu0      = mu;                           % only used for 'metallic'

fpath = [go('mu'),'/DMFT/Emery_CDMFT'];
fname = @(n) [fpath,'/CDMFT_Emery_K_U',num2str(U),'_mu',num2str(mu),'_T',num2str(T),'_',num2str(n),'_SEeta2pole.mat'];
N2 = 1;                                  % run counter
if N2 == 1
   switch startHyb
      case 'metallic'
         % U = 0 lattice step at mu0
         Rhoin = latticeK(ocont,mu0,P,zeros(Noc,Nc),latopt);      % Noc x 4

      case 'hubbardI'
         % cheaper grid for the seed iterations only
         latopt_seed = {'Nk',64,'eta',0.05,'SEeta',SEeta(ocont)};
         [SE0,nd0] = hiSeed(ocont,mu,P,U,T,latopt_seed,1.0);
         Rhoin = latticeK(ocont,mu,P,SE0,latopt);                 % Noc x 4

      otherwise
         error('ERR: startHyb must be ''metallic'' or ''hubbardI''.');
   end
   RhoV2_arr = [];  nd = [];
else
   load(fname(N2-1),'Rhoin','nd');
   Rhoin = Rhoin(:,:,end);  RhoV2_arr = [];
end

%% Step 3. Impurity Hamiltonian (K basis) and NRG parameters
% H_imp = (ed-mu)*N + U*sum_i n_i,up n_i,dn ; Wilson chains attach to FK = U*FF
[Hmu,HU,~,A0,~,ZF,~,~,~,FK,NK] = Emery_Plaquette_U1xSU2(U,mu,P.ed,'-iso2HU');

Gdep = [1 2 2 3];  Fdep = Gdep;          % diagonal self-energy; (pi,0) and (0,pi) identical
inrg = 1:Nc;                             % interleave the four independent chains

Lambda  = 4;
nz      = 2;
nrgdata = [getenv('SCRATCH_DIR'),'/Emery_K'];
Nkeep   = 5000;
NKEEP   = 3*Nkeep*ones(1,3);
Etrunc  = 9;

%% Step 4. DMFT cycle
Nmax  = 40;
iscvg = false;
W1_hist = zeros(3,0);                    % rows: K = (0,0), (pi,0)=(0,pi), (pi,pi)
while (iscvg == false) && (size(Rhoin,3) < Nmax)
   % 4.1 impurity: four diagonal channels -> SE(:,K)
   [ocont_imp,SE,nrgdataz] = impSE(Hmu,HU,A0,FK,ZF,Rhoin(:,:,end),T,Lambda,nz,Nkeep,nrgdata, ...
      'inrg',inrg,'Gdep',Gdep,'Fdep',Fdep,'broaden','adap','numCorr',3, ...
      'getAcont',optAcont, ...
      'doZLD',{'emin',1e-33,'minTE',1e-8}, ...
      'NRG_SL',{'Etrunc',Etrunc,'NKEEP',NKEEP}, ...
      'SEtrick',{'iSEmax',-1e-8});
   if numel(ocont_imp) ~= Noc || any(abs(ocont_imp(:)-ocont(:)) > 1e-12)
      error('ERR: impSE frequency grid differs from the lattice grid; check optAcont.');
   end
   SE(:,[2 3]) = repmat(mean(SE(:,[2 3]),2),1,2);    % enforce (pi,0) = (0,pi)
   nd = cat(2,nd,getEpVal(nrgdataz,sum(NK))/Nc);    % Cu filling per site

   % 4.2 lattice step in the K basis
   RhoV2 = latticeK(ocont,mu,P,SE,latopt);
   RhoV2_arr = cat(3,RhoV2_arr,RhoV2);

   % 4.3 convergence: Wasserstein-1 distance (input of this iteration vs lattice output)
   W1 = zeros(3,1);  ch = [1 2 4];
   for j = 1:3
      W1(j) = W1fun(ocont(:), Rhoin(:,ch(j),end), RhoV2(:,ch(j)));
   end
   W1_hist = cat(2,W1_hist,W1);
   fprintf('iter %d: nd = %.4f | W1 (0,0) = %.2e, (pi,0) = %.2e, (pi,pi) = %.2e\n', ...
           size(W1_hist,2), nd(end), W1);
   if size(W1_hist,2) >= Ncons
      iscvg = all(all(W1_hist(:,end-Ncons+1:end) < tolW));
   end

   % 4.4 update hybridization at fixed mu (x = 3: iterations along dim 3; mixing only,
   %     convergence decided above)
   [~,Rho_new] = updateHyb(Rhoin,RhoV2_arr,3,'cvgth',0,'Broyden',2,'amix',0.9,'ocont',ocont,'-v');
   Rhoin = cat(3,Rhoin,max(Rho_new,0));

   save(fname(N2),'-v7.3');
end

%% Step 5. Spectra and occupations from the converged cluster self-energy
% removeSEeta = true: oxygens without SEeta (physical U_p = 0 spectra; the bare O
% poles then need the finite eta). false: the lattice the loop used.
if removeSEeta %#ok<UNRCH>
   SEo = @(w) [];
else
   SEo = SEeta;
end
[nd_latt, np_latt, ntot, AK, Ap] = emery_cdmftK_spectra(ocont,mu,P,SE,T,'Method','adaptive','eta',0.02,'SEeta',SEo(ocont));
fprintf('mu = %.4f: nd(imp) = %.4f, nd(latt) = %.4f | np = %.4f | ntot = %.4f\n', ...
        mu, nd(end), nd_latt, np_latt, ntot);

% spectra on a linear grid (Sigma_K interpolated); spacing eta/2 (5001 points),
% adaptive k~-integration with RelTol = 1e-3 for the plot
eta = 0.02;  Wl = 25;
wf  = linspace(-Wl,Wl,4*Wl/eta+1).';
SEf = interp1(ocont(:), SE, wf, 'linear');
[~,~,nt_l,AKf,Apf] = emery_cdmftK_spectra(wf,mu,P,SEf,T,'Method','adaptive','RelTol',1e-3,'eta',eta,'SEeta',SEo(wf),'-v');
Adf = mean(AKf,2);                                  % local Cu spectral function per site
fprintf('int Ad = %.4f, int Ap = %.4f (should be ~1, ~1)\n', trapz(wf,Adf), trapz(wf,Apf));

figure; hold on; box on;
plot(wf, Adf, 'LineWidth', 1.5);
plot(wf, Apf, 'LineWidth', 1.5);
plot(wf, Adf + 2*Apf, 'k', 'LineWidth', 1.0);
xline(0, '--', 'Color', [0.5,0.5,0.5]);
xlim([-20,10]); xlabel('\omega'); ylabel('A(\omega) per spin');
legend({'Cu d_{x^2-y^2}', 'O p (avg. over all O)', 'total'}, 'Location', 'best');
title(sprintf('Emery 2x2 CDMFT: U = %g, \\mu = %g, T = %g, n_{tot} = %0.3f', U, mu, T, nt_l));
hold off

%% ------------------------------------------------------------------------
function RhoV2 = latticeK(ocont,mu,P,SEK,latopt)
% CDMFT lattice step in the K basis with a K-diagonal self-energy SEK (Noc x 4),
% asymmetric estimator Delta_K = F_K/G_K (latticeStepK_asym.m).
% Returns RhoV2_K = max(-Im Delta_K/pi,0) (Noc x 4), (pi,0)=(0,pi) enforced.
[RhoV2,~,~,~,info] = latticeStepK_asym(ocont,mu,P,SEK,latopt{:});
if isfield(info,'nfail') && info.nfail > 0
   fprintf('WRN: %d outer k~x integrals hit MaxIntervals.\n', info.nfail);
end
if isnan(info.offK)
   fprintf('lattice step: %.1f s, identity residual = %.1e\n', info.time, max(info.idErr));
else
   fprintf('lattice step: %.1f s, off-diagonal in K = %.1e, identity residual = %.1e\n', ...
           info.time, info.offK, max(info.idErr));
end
end

function [SE0,nd0] = hiSeed(ocont,mu,P,U,T,latopt,nd0)
% Hubbard-I seed with n_d determined self-consistently at fixed mu.
% Sigma_HI(z; n_d) is local -> identical on all four K channels.
fw   = 1./(exp(ocont(:)/T)+1);           % ocont is measured from mu
eta0 = 0.05;  z = ocont(:) + 1i*eta0 + mu;
for it = 1:30
   Gat = (1-nd0/2)./(z-P.ed) + (nd0/2)./(z-P.ed-U);
   SE0 = repmat(z - P.ed - 1./Gat, 1, 4);
   [~,~,G] = latticeStepK_asym(ocont,mu,P,SE0,latopt{:});
   AK = -imag(G)/pi;
   nd_new = 2*mean(trapz(ocont(:), fw.*AK));
   fprintf('HI seed: it %2d  nd_in = %.4f  nd_out = %.4f\n', it, nd0, nd_new);
   if abs(nd_new - nd0) < 1e-3, break; end
   nd0 = 0.5*nd0 + 0.5*nd_new;
end
end
