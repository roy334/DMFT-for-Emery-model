%% Single site DMFT for the Emery (three-band Hubbard) model
% Oxygens integrated out (exact, U_p = 0):
%   G0eff^-1(z,k) = z - ed - W(z,k),   z = w + mu,
%   W(z,k) = w(k) (z - h_pp(k))^{-1} w^dag(k)          (emeryW.m)
% Single-site DMFT: Sigma(k,z) = Sigma(z) on the Cu d orbital.
%   G_loc(w)  = (1/pi^2) int_[0,pi]^2 d^2k  1/(w + mu - ed - W(w+mu,k) - Sigma(w))
%   Delta(w)  = F(w)/G_loc(w),  F = (1/pi^2) int d^2k  W/(w + mu - ed - W - Sigma)
%               (asymmetric estimator; = w + mu - ed - Sigma - 1/G_loc exactly)
%   Oxygen broadening: W(z,k) = w (z - SEeta(w) - h_pp)^{-1} w^dag with the causal
%   two-pole SEeta (Kramers-Kronig exact), in the lattice step only; it can be
%   removed after convergence for the physical oxygen spectra (removeSEeta).
% NOTE: no 1D DoS integration is possible here (W depends on k and w), so the
% lattice step is a 2D k-integration for every frequency (emery_lattice_asymest.m).

clear
num_threads_SL(8);

%% Step 1: Model parameters (Sordi et al., PRB 111, 045117)

P = struct('ed',0,'ep',8,'tpd',1.5,'tpp',1,'tpp2',1);   % oxygen diagonal: ep - 2tpp + 2tpp2 cos k

% Lattice step: emery_lattice_asymest.m (2D k-integration with the asymmetric
% estimator). No static Cu-Cu hopping: the impurity level is ed - mu, and nothing
% is subtracted from Delta (W(z,k) = O(1/z) has no static part).
% Broadening of the lattice step: causal two-pole O self-energy (U_p = 0 oxygens
% have none of their own, so their nearly flat bands give delta-like spikes in
% Gamma(w) that NRG cannot resolve):
%   SEeta(z) = etamx*wb*z/((z + i wc)(z + i wb)),   poles at -i wc, -i wb (causal, KK exact)
%   -Im SEeta(w) = etamx*wb*(wc+wb)*w^2/((w^2+wc^2)(w^2+wb^2)):
%       0 at w = 0, max etamx*wb/(wb+wc) = 0.24 at |w| = sqrt(wc*wb) = 10, ~150/w^2 beyond wb
% It enters the oxygen propagator inside W only (zO = z - SEeta(w)); the Cu part
% zeta = z - ed - Sigma_d is untouched, so the estimator needs no subtraction.
% etaLat: small constant regulator for integral2 (all states), z = w + i*etaLat + mu.
etamx  = 0.3;  wc = 5;  wb = 20;
SEeta  = @(w) etamx*wb*w(:)./((w(:)+1i*wc).*(w(:)+1i*wb));
etaLat = 1e-4;

removeSEeta = true;   % after convergence: occupations/spectra without SEeta (physical oxygens)


%% Step 2: Impurity hamiltonian, initial hybridization and NRG parameters

U  = 11.1;       % Cu interaction strength
T  = 0.05;       % temperature
mu = 10;         % fixed chemical potential

% Choice of starting hybridization (only used if Nrun == 0):
%   'metallic' : non-interacting hybridization (Sigma = 0), evaluated at mu0
%   'hubbardI' : Hubbard-I self-energy, n_d determined self-consistently at fixed mu
%   'hartree'  : static Sigma_dd = U*n_d/2 - i*Gamma0 (Hartree shift plus a constant
%                imaginary part), n_d determined self-consistently at fixed mu.
%                This is the starting guess of Vucicevic & Zitko, arXiv:2604.08085,
%                Sec. III.B, step 1 (they report that Sigma_dd = 0 converges poorly).

Lambda  = 4;     % discretization parameter
nz      = 2;     % number of z-shifts
nrgdata = [getenv('SCRATCH_DIR'),'/'];   % path to save NRG data (HDD mode)
Nkeep   = 5000;  % kept states at each step of iterative diagonalization
NKEEP   = 3*Nkeep*ones(1,3);   % kept states for the first three chain sites
Etrunc  = 10;    % truncation energy
dz      = 1e-4/nz;   % finite-difference step of doZLD's differential Hamiltonian
                     % (default 0.01/nz); smaller -> avoids H1/H2 size
                     % mismatch. H1 from rep1TE, H2 from rep2TE in doZLD

startHyb = 'hubbardI';
mu0      = mu;   % reference mu for 'metallic' (e.g. 8.65: U = 0, n_tot = 5)
Gamma0   = 0.5;  % constant broadening for 'hartree' (paper: "moderate", no value given)

Nrun = 0; % Nrun = 0: no previous run, set up the initial hybridization.
          % Nrun > 0: load the hybridization of run Nrun-1 and continue.

fname = @(n) [go('mu'),'/DMFT/Emery_code/Emery_integrate2_implementation/Emery_single_site/Ex_single_site_Emery_U_',num2str(U), ...
              '_mu_',num2str(mu),'_T_',num2str(T),'_ep_',num2str(P.ep),'_tpd_',num2str(P.tpd),'_tpp2_',num2str(P.tpp2), '_',startHyb,'_',num2str(n),'_SEeta2pole_Lambda_',num2str(Lambda),'_nz_',num2str(nz),'_.mat'];

if Nrun == 0

   ocont = getAcont(0,0,0,0);

   switch startHyb
      case 'metallic'
         % U = 0 lattice step at mu0 -> non-interacting hybridization
         Rhoin = emery_lattice_asymest(ocont,mu0,P,zeros(numel(ocont),1),'eta',etaLat,'SEeta',SEeta(ocont));

      case 'hubbardI'
         % Hubbard-I self-energy with n_d determined self-consistently at fixed mu
         eta0 = 0.05;
         z    = ocont(:) + 1i*eta0 + mu;
         fw   = 1./(exp(ocont(:)/T)+1);          % ocont is measured from mu
         n0   = 1;                               % starting guess
         for itHI = 1:30
            Gat = (1-n0/2)./(z-P.ed) + (n0/2)./(z-P.ed-U);
            SE0 = z - P.ed - 1./Gat;
            [Rhoin,~,Gl] = emery_lattice_asymest(ocont,mu,P,SE0,'eta',etaLat,'SEeta',SEeta(ocont));
            n_new = 2*trapz(ocont(:), fw.*(-imag(Gl(:))/pi));   % Cu filling, both spins
            fprintf('HI seed: it %2d  nd_in = %.4f  nd_out = %.4f\n', itHI, n0, n_new);
            if abs(n_new - n0) < 1e-3, break; end
            n0 = 0.5*n0 + 0.5*n_new;
         end
         if abs(n_new - n0) >= 1e-3
            warning('HI seed: n_d not converged (last change %.2e)', abs(n_new - n0));
         end

      case 'hartree'
         % static Hartree self-energy plus constant broadening, n_d self-consistent at fixed mu
         fw   = 1./(exp(ocont(:)/T)+1);          % ocont is measured from mu
         n0   = 1;                               % starting guess
         for itH = 1:30
            SE0 = (U*n0/2 - 1i*Gamma0)*ones(numel(ocont),1);
            [Rhoin,~,Gl] = emery_lattice_asymest(ocont,mu,P,SE0,'eta',etaLat,'SEeta',SEeta(ocont));
            n_new = 2*trapz(ocont(:), fw.*(-imag(Gl(:))/pi));   % Cu filling, both spins
            fprintf('Hartree seed: it %2d  nd_in = %.4f  nd_out = %.4f\n', itH, n0, n_new);
            if abs(n_new - n0) < 1e-3, break; end
            n0 = 0.5*n0 + 0.5*n_new;
         end
         if abs(n_new - n0) >= 1e-3
            warning('Hartree seed: n_d not converged (last change %.2e)', abs(n_new - n0));
         end

      otherwise
         error('ERR: startHyb must be ''metallic'', ''hubbardI'' or ''hartree''.');
   end

   Rhoin = real(Rhoin(:));
   RhoV2_arr = [];
   nd = [];

else
   % continue from a previous run with the same parameters
   load(fname(Nrun-1),'ocont','Rhoin');
   Rhoin = Rhoin(:,end);
   RhoV2_arr = [];
   nd = [];
end

% create impurity hamiltonian and chain operators: single Cu d orbital,
% level ed - mu (t_cl = ed: no static hopping in the Emery model)
[Hmu,HU,~,A0,FF,ZF,SF,EF,NF] = SIAM('Acharge,SU2spin',U,mu,P.ed,'-iso2HU');




minRho = 0;  % Do not clip off the edges of the hybridization, instead, fix a large window of [-emax,+emax]
minTE = 1e-3;   % tolerence level in doZLD; if hybridization weight inside an interval is less than minTE, the interval is merged with its's neighbor
Emax = 50;       % fixed outer boundary of the discretization grid (beyond all bands)
Emin = 1e-12;  % inner edge of the discretization grid
Nchain = ceil(-2*log(T/(100*Emax))/log(Lambda)) + 5;    % chain starts at the grid top, and fixed for all iterations. Extending it to 2 decades below T for good measure.


fprintf('Wilson chain length fixed to Nchain = %d (Emax = %.1f)\n', Nchain, Emax);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The commented lines below are not used, in favor of the fixed Emax, and minRho = 0

%Gin0   = Rhoin(:,end);
%Dhyb   = 1.5*max(abs(ocont(Gin0 > 1e-2*max(Gin0))));
%Nchain = ceil(-2*log(T/(100*Dhyb))/log(Lambda)) + 4;
%fprintf('Wilson chain length fixed to Nchain = %d (Dhyb = %.2f)\n', Nchain, Dhyb);

% Wilson chain length, FIXED for the entire run (a chain length that changes
% between iterations adds noise and slows convergence). Dhyb = largest |w| with
% Gamma > 1% of its maximum in the starting hybridization (same criterion as
% doZLD's 'Initial repE'), times a margin of 1.5 for band edges that move outwards
% during the loop (costs ~1 extra site). The chain reaches ~2 decades below T.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 3: DMFT self-consistency cycle (fixed mu)

N = 100;         % maximum number of iterations
Niter = 0;
iscvg = false;

% Note because of sharp features of the hybridization function, we will use
% the Wasserstein norm instead of the 'cvgth' criteria in the update_hyb
% function

tolW    = 5e-3;   % threshold for the normalized W1 in units of t_pp
W1fun   = @(w,a,b) trapz(w, abs(cumtrapz(w,a)/trapz(w,a) - cumtrapz(w,b)/trapz(w,b)));
ncons   = 0;      % number of consecutive iterations with W1 < tolW
W1_hist = [];     % Wasserstein history, saved with the workspace
tIter = [];

tDMFT = tic;                      % total wall time of the DMFT loop

while size(Rhoin,2) < N
    tLoop = tic;
    if iscvg == true
        disp('Convergence criteria met, exiting DMFT loop')
        break
    end


    % Step 3.2: NRG impurity solver -> self-energy
    [ocont,SE,nrgdataz] = impSE(Hmu,HU,A0,FF,ZF,Rhoin(:,end),T,Lambda,nz,Nkeep,nrgdata, ...
                      'broaden','adap','numCorr',3, ...
                      'Nchain',Nchain, ...
                      'doZLD',{'emax',Emax,'emin',Emin,'minrho',minRho,'minTE',minTE,'dz',dz}, ...
                      'NRG_SL',{'Etrunc',Etrunc,'NKEEP',NKEEP},'SEtrick',{'iSEmax',-1e-8});
    nd = cat(2,nd,getEpVal(nrgdataz,NF));   % Cu occupation <n_d> at each iteration

    % Step 3.3: lattice step: 2D k-integration of the local Cu Green's function G
    % and of F = <W/(zeta - W)>; hybridization from the asymmetric estimator F/G
    [RhoV2,~,G] = emery_lattice_asymest(ocont,mu,P,SE,'eta',etaLat,'SEeta',SEeta(ocont));
    RhoV2_arr = cat(2,RhoV2_arr,real(RhoV2(:)));

    % Wasserstein-1 distance between input and output hybridization (both normalized
    % to unit weight):  W1 = int dw |C_in(w) - C_out(w)|,  C(w) = int_{-inf}^{w} Gamma / int Gamma
    W1 = W1fun(ocont(:), Rhoin(:,end), real(RhoV2(:)));
    W1_hist = cat(2, W1_hist, W1);

    fprintf('iter %d: nd = %.4f | W1 = %.3e t_pp\n', Niter+1, nd(end), W1);

    % Step 3.4: update hybridization at fixed mu (pointwise convergence criterion)
    % amix = 0.5 (was 0.9): damps the Broyden overshoots seen at sharp features
    
    [~,Rhoin_new] = updateHyb(Rhoin,RhoV2_arr,2,'cvgth',0,'Broyden',2,'amix',0.5); % Note that here, iscvg is not needed because of the Wasserstein norm

    Rhoin_new = max(Rhoin_new, 0);       % a hybridization function must be >= 0 (set by hand)
    Rhoin = cat(2,Rhoin,Rhoin_new);

    % convergence: W1 below tolW in 5 consecutive iterations
    if W1 < tolW
        ncons = ncons + 1; 
    else 
        ncons = 0; 
    end

    iscvg = (ncons >= 5);

    % save intermediate data
    save(fname(Nrun),'-v7.3');
    tIter = cat(2,tIter,toc(tLoop));                          
    save(fname(Nrun),'tIter','-append');
    Niter = Niter+1;
    disp(['Finished iteration number:',num2str(Niter),' (',num2str(tIter(end),'%.1f'),' s)']);
    
end

tTotal = toc(tDMFT);
fprintf('DMFT loop: %d iterations, total time %.1f s (%.2f h), converged = %d\n', ...
        Niter, tTotal, tTotal/3600, iscvg);
save(fname(Nrun),'tTotal','-append');

%% Step 4: Occupations from the converged self-energy
% n_d from the lattice (check against the impurity value nd(end)),
% n_p per O orbital from the exact oxygen Green's function, n_tot = n_d + 2 n_p.
% removeSEeta = true: oxygens without SEeta (physical U_p = 0 spectra; integral2
% then needs the finite eta = 0.02 for the bare O poles). false: the lattice of the loop.

if removeSEeta
    SEo = @(w) []; 
else
    SEo = SEeta; 
end


[nd_latt, np, ntot] = emery_occupations(ocont,mu,P,SE,T,'eta',0.02,'SEeta',SEo(ocont));
fprintf('mu = %.4f: nd(imp) = %.4f, nd(latt) = %.4f, np = %.4f, ntot = %.4f\n', ...
        mu, nd(end), nd_latt, np, ntot);


%% Step 5:Optional--> Interpolate over a linear grid to get the occupation and the spectral sum rule

eta=0.02;  W=25;
wf = linspace(-W,W,8*W/eta+1).';
SEf = interp1(ocont, SE, wf, 'linear');
[nd_l, np_l, nt_l, Ad, Ap] = emery_occupations(wf, mu,P,SEf,T,'eta',eta,'SEeta',SEo(wf));
Atot = Ad+2*Ap;
fprintf('int Ad = %.4f, int Ap = %.4f, int Atot = %.4f (should be ~1, ~1, ~3)\n',trapz(wf,Ad),trapz(wf,Ap),trapz(wf,Atot));

figure; hold on; box on;
plot(wf, Ad, 'LineWidth', 1.5);
plot(wf, Ap, 'LineWidth', 1.5);
plot(wf, Atot, 'k', 'Linewidth', 1.0);
xline(0, '--', 'Color', [0.5,0.5,0.5]);
xlim([-20,10]);
xlabel('\omega [t_{pp}]');
ylabel('A(\omega) per spin');
legend({'Cu d_{x^2-y^2}', 'O p_x = p_y', 'total'}, 'Location', 'best');
title(sprintf('Emery single-site DMFT: U = %g, \\mu=%g, T= %g, n_{tot}=%0.3f',U,mu, T, nt_l));
hold off


%% Diagnostic tests:

diagnostics = false;

if diagnostics


    % ---------- settings ----------
    [~,i0] = max(Rhoin(:,end) .* (ocont(:) < 0));   w0 = ocont(i0);   % spike position (~ -7.7)
    dw   = 0.6;                                     % half-width of the window around the spike
    etas = [3e-3 1e-2 3e-2 1e-1];                   % constant broadenings to scan
    nw   = 4801;                                    % spacing 2*dw/(nw-1) = 2.5e-4 < min(etas)/4
    wsub = linspace(w0-dw, w0+dw, nw).';
    SEs  = interp1(ocont, SE, wsub, 'linear');      % converged Sigma_d on the fine grid

    % ---------- lattice step for each eta ----------
    Gam = zeros(nw, numel(etas));  Gloc = zeros(nw, numel(etas));
    for k = 1:numel(etas)
        [Gam(:,k), ~, Gloc(:,k)] = emery_lattice_asymest(wsub, mu, P, SEs, 'eta', etas(k));
        fprintf('eta = %g done\n', etas(k));
    end

    % ---------- height, width, weight ----------
    height = zeros(size(etas)); fwhm = height; weight = height; wpred = height;
    for k = 1:numel(etas)
        g  = Gam(:,k);
        bg = interp1([wsub(1) wsub(end)], [g(1) g(end)], wsub);   % linear background under the peak
        gp = g - bg;
        [height(k), ip] = max(gp);
        above = find(gp >= height(k)/2);                          % FWHM from the half-maximum crossings
        fwhm(k)   = wsub(above(end)) - wsub(above(1));
        weight(k) = trapz(wsub, gp);
        % prediction: Gamma ~ (1/pi) Im(1/G) near the zero of Re G  ->  weight 1/|dReG/dw|
        reG = real(Gloc(:,k));
        iz  = find(reG(1:end-1).*reG(2:end) <= 0);               % sign changes of Re G
        if ~isempty(iz)
            [~,m] = min(abs(wsub(iz) - wsub(ip)));  iz = iz(m);
            a = (reG(iz+1) - reG(iz)) / (wsub(iz+1) - wsub(iz));
            wpred(k) = 1/abs(a);
        else
            wpred(k) = NaN;
        end
        fprintf('eta = %-6g  height = %8.3f  FWHM = %.4f  weight = %.4f  predicted 1/|a| = %.4f\n', ...
            etas(k), height(k), fwhm(k), weight(k), wpred(k));
    end

% ---------- plots ----------
    figure;
    subplot(1,3,1); plot(wsub, Gam, 'LineWidth', 1.2);
    xlabel('\omega [t_{pp}]'); ylabel('\Gamma(\omega)'); legend(compose('\\eta = %g', etas));
    subplot(1,3,2); loglog(etas, height, 'o-', etas, height(1)*etas(1)./etas, 'k--');
    xlabel('\eta'); ylabel('peak height'); legend('data','\propto 1/\eta');
    subplot(1,3,3); semilogx(etas, weight, 'o-', etas, wpred, 's--');
    xlabel('\eta'); ylabel('integrated weight'); legend('\int(\Gamma - background)','1/|\partial_\omega Re G|');


%% Relative change in hybridization function

    nit = size(RhoV2_arr,2);
    j = nit; Gin = Rhoin(:,j); r = abs(RhoV2_arr(:,j) - Gin) ./ (Gin + 1e-3*max(Gin));
    figure; semilogy(ocont, r); hold on; yline(1e-2,'r--','1%'); xlim([-15 15]); xlabel('\omega'); ylabel('relative change'); hold off
end

%% Step 7: Comparing metallic vs Hubbard-I starts, if present:

comp_start = false;

if comp_start

    % ---------- files ----------
    f = {'Ex_single_site_Emery_U_12_mu_10_T_0.05_ep_2.8_tpd_2_tpp2_0.2_metallic_0.mat', ...
        'Ex_single_site_Emery_U_12_mu_10_T_0.05_ep_2.8_tpd_2_tpp2_0.2_hubbardI_0.mat'};
    lab = {'metallic start', 'Hubbard-I start'};

    % ---------- spectra on a common linear grid ----------
    eta = 0.02;  W = 25;
    wf  = linspace(-W, W, 4*2*W/eta + 1).';
    Ad = zeros(numel(wf),2);  Ap = Ad;  ntot = zeros(1,2);  ndimp = ntot;
    for s = 1:2
        S   = load(f{s}, 'ocont','SE','mu','P','T','nd');
        SEf = interp1(S.ocont, S.SE, wf, 'linear');
        [~, ~, ntot(s), Ad(:,s), Ap(:,s)] = emery_occupations(wf, S.mu, S.P, SEf, S.T, 'eta', eta);
        ndimp(s) = S.nd(end);
    end
    Atot = Ad + 2*Ap;

    % ---------- 3-panel figure ----------
    figure('Position', [100 100 1400 400]);
    ttl = {'Cu d_{x^2-y^2}', 'O p_x (= p_y)', 'total'};
    dat = {Ad, Ap, Atot};
    for k = 1:3
        subplot(1,3,k); hold on; box on;
        plot(wf, dat{k}(:,1), 'LineWidth', 1.2);
        plot(wf, dat{k}(:,2), '--', 'LineWidth', 1.2);
        xline(0, ':', 'Color', [0.5 0.5 0.5]);
        xlim([-20 10]); xlabel('\omega [t_{pp}]'); ylabel('A(\omega) per spin'); title(ttl{k});
        legend(lab, 'Location', 'best'); hold off;
    end

    fprintf('%-16s  nd(imp) = %.4f   ntot = %.4f   delta = %.4f\n', lab{1}, ndimp(1), ntot(1), 5-ntot(1));
    fprintf('%-16s  nd(imp) = %.4f   ntot = %.4f   delta = %.4f\n', lab{2}, ndimp(2), ntot(2), 5-ntot(2));

    % Maximum difference and fillings

    fprintf('max|dAd| = %.2e, max|dAp| = %.2e, max|dAtot| = %.2e\n', ...
        max(abs(diff(Ad,1,2))), max(abs(diff(Ap,1,2))), max(abs(diff(Atot,1,2))));
    fprintf('ntot: %.4f vs %.4f;  nd(imp): %.4f vs %.4f\n', ntot, ndimp);



    %% Step 8 Comparing linear vs log grid

    % ---------- run to analyse ----------
    S = load('Ex_single_site_Emery_U_12_mu_10_T_0.05_ep_2.8_tpd_2_tpp2_0.2_hubbardI_0.mat', ...
         'ocont','SE','mu','P','T','nd');

    eta = 0.02;  W = 25;                                   % same broadening and window for both grids

    % ---------- logarithmic grid (DMFT grid ocont, restricted to |w| <= W) ----------
    wl  = S.ocont(abs(S.ocont) <= W);  wl = wl(:);
    SEl = interp1(S.ocont, S.SE, wl, 'linear');
    [ndl, npl, ntl, Adl, Apl] = emery_occupations(wl, S.mu, S.P, SEl, S.T, 'eta', eta);

    % ---------- linear grid, spacing eta/4 ----------
    wf  = linspace(-W, W, 4*2*W/eta + 1).';
    SEf = interp1(S.ocont, S.SE, wf, 'linear');
    [ndf, npf, ntf, Adf, Apf] = emery_occupations(wf, S.mu, S.P, SEf, S.T, 'eta', eta);

    % ---------- table ----------
    fprintf('%-7s %6s %8s %8s %8s %9s %8s %8s %8s\n', ...
        'grid','N','int Ad','int Ap','int Atot','nd(latt)','np','ntot','delta');
    fprintf('%-7s %6d %8.4f %8.4f %8.4f %9.4f %8.4f %8.4f %8.4f\n', 'log', numel(wl), ...
        trapz(wl,Adl), trapz(wl,Apl), trapz(wl,Adl+2*Apl), ndl, npl, ntl, 5-ntl);
    fprintf('%-7s %6d %8.4f %8.4f %8.4f %9.4f %8.4f %8.4f %8.4f\n', 'linear', numel(wf), ...
        trapz(wf,Adf), trapz(wf,Apf), trapz(wf,Adf+2*Apf), ndf, npf, ntf, 5-ntf);
    fprintf('nd(imp) = %.4f   |nd(latt)-nd(imp)|: log %.1e, linear %.1e\n', ...
        S.nd(end), abs(ndl-S.nd(end)), abs(ndf-S.nd(end)));

    % ---------- where the difference comes from: A_p at the oxygen peak ----------
    w_dec = S.P.ep - 2*S.P.tpp + 2*S.P.tpp2 - S.mu;        % decoupled p_x line (-8.8 here)
    figure; plot(wf, Apf, '-', wl, Apl, 'o-', 'MarkerSize', 3); hold on;
    xline(w_dec, ':', 'decoupled p_x line'); xlim([w_dec-1.5 w_dec+1.5]);
    legend('linear grid','log grid (ocont)'); xlabel('\omega [t_{pp}]'); ylabel('A_p(\omega)'); hold off


    %% Step 9: comparing relative change in hybridization between metallic and Hubbard-I starts

    f   = {'Ex_single_site_Emery_U_12_mu_10_T_0.05_ep_2.8_tpd_2_tpp2_0.2_metallic_0.mat', ...
       'Ex_single_site_Emery_U_12_mu_10_T_0.05_ep_2.8_tpd_2_tpp2_0.2_hubbardI_0.mat'};
    lab = {'metallic start', 'Hubbard-I start'};

    figure('Position', [100 100 1200 420]);
    ax = gobjects(1,2);
    for s = 1:2
        S   = load(f{s}, 'ocont', 'Rhoin', 'RhoV2_arr');
        j   = size(S.RhoV2_arr, 2);                    % last completed iteration
        Gin = S.Rhoin(:,j);                            % its input hybridization
        r   = abs(S.RhoV2_arr(:,j) - Gin) ./ (Gin + 1e-3*max(Gin));

        ax(s) = subplot(1,2,s);
        semilogy(S.ocont, r); hold on;
        yline(1e-2, 'r--', '1%');
        xlim([-15 15]); xlabel('\omega [t_{pp}]'); ylabel('relative change');
        title(sprintf('%s, iteration %d', lab{s}, j)); hold off;

        [m, i] = max(r(abs(S.ocont) <= 15));  wsel = S.ocont(abs(S.ocont) <= 15);
        fprintf('%-16s iteration %d: max relative change = %.2f %% at w = %.3f\n', lab{s}, j, 100*m, wsel(i));
    end
    linkaxes(ax, 'y');                                 % same y-range in both panels

    f   = {'Ex_single_site_Emery_U_12_mu_10_T_0.05_ep_2.8_tpd_2_tpp2_0.2_metallic_0.mat', ...
       'Ex_single_site_Emery_U_12_mu_10_T_0.05_ep_2.8_tpd_2_tpp2_0.2_hubbardI_0.mat'};
    lab = {'metallic start', 'Hubbard-I start'};
    win = 15;                                         % frequency window |w| <= win

    figure('Position', [100 100 1200 420]);
    ax = gobjects(1,2);
    for s = 1:2
        S  = load(f{s}, 'ocont', 'Rhoin', 'RhoV2_arr');
        m  = abs(S.ocont(:)) <= win;  w = S.ocont(m);
        sets = {S.Rhoin(m,:), S.RhoV2_arr(m,:)};     % input history, output history
        ax(s) = subplot(1,2,s); hold on;
        mk = {'o-','s-'};  nm = {'\Gamma_{in}: j \rightarrow j+1', '\Gamma_{out}: j \rightarrow j+1'};
        for q = 1:2
            Gh = sets{q};  nj = size(Gh,2) - 1;
            cmax = zeros(1,nj);  cint = zeros(1,nj);
            for j = 1:nj
                G0 = Gh(:,j);  G1 = Gh(:,j+1);
                cmax(j) = max(abs(G1 - G0) ./ (G0 + 1e-3*max(G0)));        % worst local relative change
                cint(j) = trapz(w, abs(G1 - G0)) / trapz(w, G0);           % integrated relative change
            end
            semilogy(1:nj, cmax, mk{q}, 'MarkerSize', 3, 'DisplayName', [nm{q} ', max_\omega']);
            semilogy(1:nj, cint, [mk{q}(1) '--'], 'MarkerSize', 3, 'DisplayName', [nm{q} ', integrated']);
        end
        set(gca, 'YScale', 'log'); yline(1e-2, 'r--', '1%', 'HandleVisibility', 'off');
        xlabel('iteration j'); ylabel('relative change between iterations');
        title(lab{s}); legend('show', 'Location', 'northeast'); box on; hold off;
    end
    linkaxes(ax, 'y');
end    