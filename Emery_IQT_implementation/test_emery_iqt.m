%% Check emery_lattice_iqt / emery_occupations_iqt against the integral2 versions
% Run once after installing IQT. Loads a saved two-impurity run (any Nrun file of
% Ex_single_site_Emery_twoimp.m) and compares both lattice steps on a subset of
% frequencies at eta = 0.02, where integral2 is reliable, then shows timings.

fileRun = '/Users/re98diy/Documents/MATLAB/qspace4u-qspace-v4-osx15-sequoia-5388fa025a15/MuNRG-main/DMFT/Emery_code/Emery_integrate2_implementation/Emery_two_impurity_single_site/Emery_twoimp_Ud_12_Up_4_mu_12.2_T_0.05_ep_-1.2_epDC_-4_tpd_2_tpp2_0.2_hubbardI_hartree_0_const_broadening.mat';            % <- full path of a saved .mat file of the two-impurity loop
S = load(fileRun, 'ocont', 'SEd', 'SEp', 'mu', 'P', 'T');

N    = numel(S.ocont);
near = find(abs(S.ocont) < 15);                                % denser where the bands are
idx  = unique([round(linspace(1, N, 60)), near(1:25:end).']);
w = S.ocont(idx);  sd = S.SEd(idx);  sp = S.SEp(idx);

% 1) lattice step: integral2 (old) vs IQT (new), same eta
tic; [R0, D0, G0, F0] = emery_lattice_twoimp(w, S.mu, S.P, sd, sp, 'eta', 0.02); t0 = toc;
tic; [R1, D1, G1, F1, info] = emery_lattice_iqt(w, S.mu, S.P, sd, sp, 'eta', 0.02); t1 = toc;
fprintf('lattice step on %d frequencies: integral2 %.1f s, IQT %.1f s\n', numel(w), t0, t1);
fprintf('max |G_new - G_old| / max|G| : Cu %.2e, O %.2e\n', max(abs(G1 - G0)) ./ max(abs(G0)));
fprintf('max |Gamma_new - Gamma_old| / max Gamma : Cu %.2e, O %.2e\n', max(abs(R1 - R0)) ./ max(R0));
fprintf('IQT: %d failed, %d roundoff; identity residual F - (zeta G - 1): %.1e, %.1e\n', ...
        info.nFailed, info.nRoundoff, info.idErr);

% 2) the same with small eta (only IQT; integral2 would struggle here)
tic; [R2, ~, ~, ~, info2] = emery_lattice_iqt(w, S.mu, S.P, sd, sp, 'eta', 1e-6); t2 = toc;
fprintf('IQT at eta = 1e-6: %.1f s, %d failed, %d roundoff, identity residual %.1e, %.1e\n', ...
        t2, info2.nFailed, info2.nRoundoff, info2.idErr);

figure; hold on; box on;
plot(w, R0(:,1), 'o', w, R1(:,1), '-', w, R2(:,1), '--');
plot(w, R0(:,2), 's', w, R1(:,2), '-', w, R2(:,2), '--');
xlim([-20 10]); xlabel('\omega'); ylabel('\Gamma(\omega)');
legend('\Gamma_d integral2 (\eta=0.02)', '\Gamma_d IQT (\eta=0.02)', '\Gamma_d IQT (\eta=10^{-6})', ...
       '\Gamma_p integral2', '\Gamma_p IQT', '\Gamma_p IQT (\eta=10^{-6})');

% 3) occupations: trapz on a linear grid (old) vs expectation values (new, no grid, no eta)
tic; [nd1, np1, nt1, ~, ~, infoN] = emery_occupations_iqt(S.ocont, S.mu, S.P, S.SEd, S.SEp, S.T); t3 = toc;
fprintf('IQT occupations (%.1f s): nd = %.5f, np = %.5f, ntot = %.5f\n', t3, nd1, np1, nt1);
fprintf('Sigma_inf from the continuation: Re Sigma_d = %.4f, Re Sigma_p = %.4f\n', infoN.SigmaInf);
