%% Plotting function to get density of states and self energies

Ud = 12; Up = 0; mu = 10; T = 0.05;
epDC = 0;
Nrun = 0;

P = struct('ed',0,'ep',2.8,'tpd',2,'tpp',1,'tpp2',0.2);
P.ep = P.ep + epDC;
startD = 'hubbardI';  startP = 'hartree';
fname = [go('mu'),'/DMFT/Emery_code/Emery_IQT_implementation/Emery_two_impurity_single_site/Emery_twoimp_Ud_',num2str(Ud), ...
              '_Up_',num2str(Up),'_mu_',num2str(mu),'_T_',num2str(T),'_ep_',num2str(P.ep),'_epDC_',num2str(epDC),'_tpd_',num2str(P.tpd), ...
              '_tpp2_',num2str(P.tpp2),'_',startD,'_',startP,'_',num2str(Nrun),'_IQT_SEeta.mat'];

% Load the required variables from the data
data  = load(fname,'G','SEd','SEp','ocont','nd','np');
ocont = data.ocont(:);  SEd = data.SEd;  SEp = data.SEp;
nd = data.nd;
np = data.np;
ntot_imp = nd+2*np;

% local spectral functions of the last lattice step (z = w + mu + i*1e-6; O includes SEeta)
Ad_log = -imag(data.G(:,1))/pi;
% Remove SEeta from the oxygen broadening now

%eta0 = 0.02; etamx = 0.3; wc = 5; wb = 20; G0 = 20;                 % as in the run
%SEeta = @(w) eta0*G0./(w(:)+1i*G0) + etamx*wb*w(:)./((w(:)+1i*wc).*(w(:)+1i*wb));

% Causal two-pole broadening (lattice step only, O orbital), Kramers-Kronig exact:
%   -Im SEeta(w) = etamx*wb*(wc+wb)*w^2 / ((w^2+wc^2)(w^2+wb^2))
etamx = 0.3;  wc = 5;  wb = 20;
SEeta = @(w) etamx*wb*w(:)./((w(:)+1i*wc).*(w(:)+1i*wb));

% Causal broadening with a Gaussian cutoff (lattice step only, O orbital):
%   -Im SEeta(w) = etamx * w^2/(w^2+wc^2) * exp(-w^2/wb^2)
%   SEeta(z)     = int de rhoEta(e)/(z - e),  rhoEta = -Im SEeta/pi   (Kramers-Kronig exact)
%etamx = 0.4;  wc = 5;  wb = 20;
%de  = 0.02;  e = (-150:de:150).';  dlt = 2*de;     % eps grid and small smoothing (<< wc)
%rhoEta = (etamx/pi)*e.^2./(e.^2+wc^2).*exp(-e.^2/wb^2);
%SEeta  = @(w) arrayfun(@(x) trapz(e, rhoEta./(x + 1i*dlt - e)), w(:));



Gp_phys = 1./(1./data.G(:,2) + SEeta(ocont));                       % removes SEeta exactly
Ap_log  = -imag(Gp_phys)/pi;

% linear grid and interpolation
Wl = 25;  dw = 1e-3;
wf = (-Wl:dw:Wl).';
Ad = interp1(ocont, Ad_log, wf, 'linear');
Ap = interp1(ocont, Ap_log, wf, 'linear');
Atot = Ad + 2*Ap;

% occupations (both spins; np per O orbital)
f  = 1./(exp(wf/T)+1);
nd_l = 2*trapz(wf, f.*Ad);
np_l = 2*trapz(wf, f.*Ap);
nt_l = nd_l + 2*np_l;
fprintf('lattice : nd = %.4f, np = %.4f, ntot = %.4f\n', nd_l, np_l, nt_l);
fprintf('impurity: nd = %.4f, np = %.4f, ntot = %.4f\n', data.nd(end), data.np(end), data.nd(end)+2*data.np(end));
fprintf('int Ad = %.4f, int Ap = %.4f, int Atot = %.4f (should be ~1, ~1, ~3)\n', ...
        trapz(wf,Ad), trapz(wf,Ap), trapz(wf,Atot));

fig1 = figure; hold on; box on;
plot(wf, Ad, 'LineWidth', 1.5);
plot(wf, Ap, 'LineWidth', 1.5);
plot(wf, Atot, 'k', 'LineWidth', 1.0);
ax = gca;
ax.XAxis.FontSize = 16;            
ax.YAxis.FontSize = 16;  
xline(0, '--', 'Color', [0.5,0.5,0.5]);
xlim([-20,10]);
xlabel('\omega [t_{pp}]','FontSize',18);
ylabel('A(\omega) per spin','FontSize',18);
legend({'Cu d_{x^2-y^2}', 'O p_x = p_y', 'total'}, 'Location', 'best','FontSize',18);
title(sprintf('U_d = %g, U_p = %g, \\mu = %g, T = %g, n_{tot} = %0.3f', ...
              Ud, Up, mu, T, nt_l),'FontSize',18);
hold off
exportgraphics(fig1, strrep(fname,'.mat','_DOS.pdf'), 'ContentType','vector');

fig2 = figure; hold on; box on;
plot(ocont, -imag(SEd), 'LineWidth', 1.5);
plot(ocont, -imag(SEp), 'LineWidth', 1.5);
ax = gca;
ax.XAxis.FontSize = 16;            
ax.YAxis.FontSize = 16;
xlim([-20,10]);
xlabel('\omega','Fontsize',18); ylabel('-Im \Sigma(\omega)','FontSize',18);
legend({'\Sigma_d', '\Sigma_p'}, 'Location', 'best','FontSize',18);
title(sprintf('U_d = %g, U_p = %g, \\mu = %g, T = %g, n_{tot} = %0.3f', ...
              Ud, Up, mu, T, nt_l),'FontSize',18);
hold off
exportgraphics(fig2, strrep(fname,'.mat','_Self_energy.pdf'), 'ContentType','vector');


%fig3 = figure; hold on; box on;
%plot(ocont, -imag(SEeta(ocont)), 'LineWidth', 1.5);
%plot(ocont, real(SEeta(ocont)), 'LineWidth', 1.5);
%xlim([-50,50]);
%yline(0, '--', 'Color', [0.5,0.5,0.5]);
%%xlabel('\omega'); ylabel('\Sigma_{\eta}(\omega)');
%legend({'-Im\Sigma_\eta', 'Re \Sigma_\eta'}, 'Location', 'best');
%hold off
%exportgraphics(fig3, 'Broadening_function.pdf', 'ContentType','vector');