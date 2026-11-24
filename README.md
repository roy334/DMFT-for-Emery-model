# DMFT for the Emery model

MATLAB codes for dynamical mean-field theory (DMFT) and cellular DMFT (CDMFT) of the
Emery (three-band Hubbard) model of the CuO2 plane, with the numerical renormalization
group (NRG) as impurity solver (MuNRG / QSpace).

Model: Cu d_{x^2-y^2} orbital (level `ed`, Hubbard `U` on Cu) and two O p orbitals per
unit cell (level `ep`), Cu-O hopping `tpd`, O-O hoppings `tpp`, `tpp2`.

## Contents

```
Emery_integrate2_implementation/
  Emery_CDMFT_cluster_momenta_basis/   2x2 Cu-plaquette CDMFT (current)
  Emery_single_site/                   single-site DMFT, one Cu impurity (current)
  Old_codes/                           earlier versions (site basis, old K basis,
                                       old single site) and diagnostic plots
Emery_IQT_implementation/
  Emery_two_impurity_single_site/      single-site DMFT with two impurities (Cu and O),
                                       lattice integrals with IQT
  test_emery_iqt.m
```

`.mat` files are DMFT results; `.pdf` / `.png` files are spectra, self-energies and
diagnostic plots.

## Methods

**Oxygens integrated out** (single-site and plaquette codes). The O orbitals are
uncorrelated (U_p = 0) and are integrated out exactly, giving an effective Cu problem
with the frequency-dependent kernel

    W(z,k) = w(k) (zO - h_pp(k))^{-1} w(k)^dag,

so that G_d(k,z)^{-1} = z - ed - W(z,k) - Sigma_d(z).

**Asymmetric estimator for the hybridization.** Delta(z) = F(z)/G(z) with
F = < W G >_k (lattice average), instead of Delta = zeta - 1/G. Since W = O(1/z) no static
term has to be subtracted. For the plaquette in the cluster-momentum basis
K = (0,0), (pi,0), (0,pi), (pi,pi) the full 4x4 matrices are multiplied and only the
diagonal elements of G and F are integrated; Delta_K is diagonal by symmetry, so the NRG
has four decoupled Wilson chains.

**Causal oxygen broadening.** The bare O states are broadened by a causal two-pole
self-energy inside W only, zO = z - Sigma_eta(z),

    Sigma_eta(w) = etamx * wb * w / ((w + i wc)(w + i wb)),   etamx = 0.3, wc = 5, wb = 20,

which vanishes at w = 0, is maximal near |w| = sqrt(wc wb) and decays as 1/w^2, and
satisfies Kramers-Kronig. It can be removed after convergence for the physical O spectra
and occupations (`removeSEeta = true`).

**Convergence** is measured by the normalized Wasserstein-1 distance between input and
output hybridization (per K channel for the plaquette).

**Lattice integration.**
- Single site: MATLAB `integral2` with `'Method','iterated'` (nested 1D adaptive
  quadrature), which resolves the narrow Fermi-surface ridge of a metal.
- Plaquette: `'Method','adaptive'` in `latticeStepK_asym.m` and `emery_cdmftK_spectra.m`,
  nested adaptive Gauss-Kronrod (7,15) over the reduced zone [0,pi]^2 with batched 4x4
  inversions (`pageinv`); `'Method','grid'` gives a fixed midpoint grid.
- Two-impurity version: IQT (`emery_lattice_iqt.m`).

The frequency loops of the plaquette lattice step and spectra are `parfor` loops.

## Main scripts

| Script | Purpose |
|---|---|
| `Emery_CDMFT_cluster_momenta_basis/CDMFT_Emery_K_loop.m` | plaquette CDMFT loop (driver) |
| `.../latticeStepK_asym.m` | lattice step, K basis, asymmetric estimator |
| `.../emery_cdmftK_spectra.m` | Cu and O spectra and occupations after convergence |
| `.../WK_matrix.m`, `.../emeryW.m` | O-mediated kernel W in the K basis / per k |
| `.../Emery_Plaquette_U1xSU2.m` | QSpace impurity Hamiltonian, U(1)xSU(2) |
| `Emery_single_site/Ex_single_site_Emery.m` | single-site DMFT loop (driver) |
| `.../emery_lattice_asymest.m` | lattice step, asymmetric estimator |
| `.../emery_occupations.m` | spectra and occupations |
| `Emery_two_impurity_single_site/Ex_single_site_Emery_twoimp_iqt.m` | two-impurity loop with IQT |

## Requirements

- MATLAB R2022a or newer (`pageinv`, `pagemtimes`), Parallel Computing Toolbox
  (`parfor`, `parfeval`)
- MuNRG and QSpace (not included), with MEX files built for the platform in use
- IQT (only for `Emery_IQT_implementation`)

## Running

Run the driver scripts from MATLAB with MuNRG and QSpace on the path. The plaquette
driver opens a local parallel pool; on a SLURM cluster it reads the allocated cores
from `SLURM_CPUS_PER_TASK`:

```
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

matlab -batch "CDMFT_Emery_K_loop"
```

Results are written to `[go('mu'),'/DMFT/Emery_CDMFT']` (plaquette).
