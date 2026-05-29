# CMB Lensing Reconstruction: GI, QE, and Joint-MAP

Masters thesis code — Cambridge Part III Physics 2025/26.

Compares three CMB lensing reconstruction methods at ultra-low noise (0.1 µK-arcmin, 0.3' beam, L_max = 12000):

- **Quadratic Estimator (QE)** — standard harmonic-space minimum-variance estimator
- **Gradient Inversion (GI)** — real-space estimator based on Hadzhiyska et al. (2019)
- **Joint MAP** — iterative maximum-a-posteriori reconstruction via `MAP_joint` in [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl)

An independent cross-check uses [LensIt](https://github.com/carronj/LensIt) (Python) for QE and MAP on the same simulations.

---

## Repository structure

```
├── run_qe_gi_wl12k.jl          main simulation pipeline (QE / GI / MAP, 500+ sims)
├── run_lensit_ul.py             LensIt external QE+MAP comparison
├── export_julia_cls.jl          export CAMB Cls to .npz for LensIt
├── fig1_reconstruction_maps.jl  Fig. 1 — κ reconstruction maps
├── plot_results.jl              all paper figures and SNR table
├── utils.jl                     module wrapper for Functions/
├── Functions/
│   ├── gradient.jl              FFT gradient helper
│   ├── gradient_inversion.jl    GI estimator variants + N0 estimators
│   ├── debias.jl                W_L transfer-function debiasing
│   ├── spectra.jl               spectrum binning utilities
│   ├── io.jl                    JLD2 I/O helpers
│   └── plotting.jl              shared plotting utilities
├── results/                     committed key figures and SNR table
└── Project.toml                 Julia package dependencies
```

---

## Setup

### Julia environment

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### CMBLensing.jl (dev mode — required)

This project uses a fork of CMBLensing.jl with two patches applied:

| File | Change |
|------|--------|
| `src/maximization.jl` | Numerical stability fixes for `MAP_joint`: regularised secant Hessian update, NaN guard on the ϕ line-search, diagnostic loglikelihood/logprior decomposition |
| `src/proj_lambert.jl` | Fix `logdet` ambiguity for `Diagonal{Real,LambertField{Map}}` |

Clone and check out the patched fork:

```bash
git clone https://github.com/louisl3grand/CMBLensing.jl
cd CMBLensing.jl && git checkout 5ddb912
```

Then add it as a dev dependency:

```julia
Pkg.develop(path="path/to/CMBLensing.jl")
```

### Python / LensIt environment

```bash
conda create -n lensit python=3.10
conda activate lensit
conda install -c conda-forge numpy h5py scipy pyfftw
pip install git+https://github.com/carronj/LensIt
```

---

## Running the pipeline

### Step 1 — QE, GI, and MAP simulations (Julia)

Runs 100 QE/GI sims and up to 500 MAP sims. Resumes from checkpoint if interrupted.

```bash
julia run_qe_gi_wl12k.jl
```

Output files written to `results/` (not committed; ~GB per noise level):
- `phi_maps_qe_gi_12000_ul.jld2` — per-sim ϕ_true, ϕ_QE, ϕ_GI
- `WL_qe_gi_12000_ul.jld2` — QE and GI transfer functions W_L
- `phi_maps_map_12000_ul_zero_a01.jld2` — per-sim ϕ_MAP
- `WL_map_12000_ul_zero_a01.jld2` — MAP transfer function and logpdf histories

### Step 2 — Export Cls for LensIt (Julia, run once)

```bash
julia export_julia_cls.jl
```

### Step 3 — LensIt QE + MAP comparison (Python)

```bash
conda run -n lensit python run_lensit_ul.py
```

### Step 4 — Figures

```bash
julia fig1_reconstruction_maps.jl   # Fig. 1 — κ maps
julia plot_results.jl               # all other paper figures
```

---

## Notes

### Simulation seeds

All simulations use `seed = 1000 + sim_index` (Julia) and `idx` (LensIt, 0-indexed). Seeds are stored alongside each simulation in the output JLD2 files for exact reproducibility.

### QE RDN0

The QE realisation-dependent N0 is computed via `N0_bias`, developed as part of [this CMBLensing.jl fork](https://github.com/louisl3grand/CMBLensing.jl/blob/master/RDN0_bias_QE.ipynb). It uses the standard DD–DS–SD+SS combination to achieve O(ε²) bias in per-realisation power mismatch.

### GI N0 methods

Three GI N0 estimators are implemented and compared:
- **fg-MC**: fixed-gradient Monte Carlo — holds the per-sim gradient fixed while drawing independent null T maps
- **RDN0**: realisation-dependent N0 using the DS+SD−SS combination with fiducial gradient variances σ_fid
- **lin-RD**: linearised realisation-dependent N0 — expands the noise around the mean (fiducial) data power

The fg-MC and lin-RD estimates are used together as the primary debiasing strategy; RDN0 is included as a cross-check.

### CMBLensing.jl preconditioner patch

`run_qe_gi_wl12k.jl` patches `Hessian_logpdf_preconditioner` at runtime (via `@eval CMBLensing`) to use the lensed power spectrum C_ℓ̃ instead of the unlensed C_ℓ. This reduces CG iterations at low noise without modifying the source.

---

## Key results (committed)

| File | Description |
|------|-------------|
| `results/fig1_gi_qe_map_gradient_ul.pdf` | κ reconstruction maps — GI, QE, MAP, ∇T, true |
| `results/fig2_mean_spectra.png` | Mean cross and auto power spectra |
| `results/fig3_sigma_panels.png` | σ(C_L) auto and cross, S4-like and UL |
| `results/fig_WL.pdf` | Empirical transfer functions W_L |
| `results/fig6_rho_L.pdf` | Correlation coefficient ρ_L vs L |
| `results/fig_covariance_correlation.png` | Bandpower correlation matrices |
| `results/figC_map_convergence.pdf` | MAP log-posterior convergence |
| `results/snr_table.txt` | SNR summary table (L = 4000–12000) |

---

## References

Hadzhiyska, B., Sherwin, B. D., Madhavacheril, M., and Ferraro, S. (2019), *Improving small-scale CMB lensing reconstruction*, Phys. Rev. D **100**, 023547. [doi:10.1103/PhysRevD.100.023547](https://link.aps.org/doi/10.1103/PhysRevD.100.023547) — gradient-inversion estimator, normalisation, and survey parameters.

Carron, J. and Lewis, A. (2017), *Maximum a posteriori CMB lensing reconstruction*, Phys. Rev. D **96**, 063510. [doi:10.1103/PhysRevD.96.063510](https://link.aps.org/doi/10.1103/PhysRevD.96.063510) — iterative MAP lensing reconstruction algorithm implemented in LensIt.

Millea, M., Anderes, E., and Wandelt, B. D. (2020), *Bayesian delensing of CMB temperature and polarization*, Phys. Rev. D **102**, 123542. [doi:10.1103/PhysRevD.102.123542](https://link.aps.org/doi/10.1103/PhysRevD.102.123542) — joint MAP algorithm implemented in CMBLensing.jl.

Lewis, A., Challinor, A., and Lasenby, A. (2000), *Efficient Normal Form Decomposition of General Inflationary Perturbations*, ApJ **538**, 473. [doi:10.1086/309179](https://doi.org/10.1086/309179) — CAMB power spectrum code used to generate input Cℓ's.

---

## AI use declaration

The AI assistant Claude was used during the development of this codebase to assist with code editing, debugging, and documentation.
