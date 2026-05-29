#!/usr/bin/env python3
"""
run_lensit_ul.py  — LensIt MAP reconstruction, ultra-low-noise case.

  0.1 µK-arcmin, 0.3' beam, Lmax=12000, 512×512 patch (θpix≈0.744')

Output (JLD2-compatible HDF5, readable by JLD2.jl):
  results/lensit/ul/WL_lensit_qe_ul.jld2         — QE transfer functions (W_qe_raw, nsims_completed)
  results/lensit/ul/phi_maps_lensit_qe_ul.jld2    — per-sim ϕ_true, ϕ_qe_raw
  results/lensit/ul/WL_lensit_map_ul.jld2         — MAP transfer functions (W_mj, nsims_completed)
  results/lensit/ul/phi_maps_lensit_map_ul.jld2   — per-sim ϕ_true, ϕ_mj

Usage:
  conda run -n lensit python run_lensit_ul.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import os
import h5py
import shutil
import time

from lensit import get_ellmat
from lensit.ffs_covs.ell_mat import ffs_alm_pyFFTW
from lensit.ffs_covs import ffs_cov
from lensit.misc.misc_utils import gauss_beam
from lensit.sims import ffs_cmbs, ffs_maps
from lensit.ffs_iterators.ffs_iterator_nufft import ffs_iterator_pertMF
from lensit.ffs_deflect import ffs_deflect
from lensit.qcinv import ffs_ninv_filt_ideal_nufft, chain_samples, opfilt_cinv

# Parameters
NlevT        = 0.1           # µK-arcmin
beamFWHM     = 0.3           # arcmin
LD_res       = 9             # 2^9 = 512 pixels
HD_res       = 9
θpix_arcmin  = 0.7438046267475303
θpix_rad     = θpix_arcmin * np.pi / (180. * 60.)
ellmin       = 100
ellmax       = 12000
Δℓ           = 150           # spectral binning (matches Julia)
N_sims       = 100
N_iter       = 200
CG_tol       = 1e-6
CG_maxiter   = 500
BFGS_MEMORY  = 20
GRAD_CONV    = 0.04          # stop early when grad_ratio < 4%

nTpix = NlevT / θpix_arcmin

# Paths
out_dir      = 'results/lensit/ul'
lib_dir_iso  = 'TEMP/lensit_ul_isocov'
lib_dir_sims = 'TEMP/lensit_ul_sims'
iter_base    = 'TEMP/lensit_ul_iter'
qe_wl_file   = f'{out_dir}/WL_lensit_qe_ul.jld2'
qe_phi_file  = f'{out_dir}/phi_maps_lensit_qe_ul.jld2'
wl_file      = f'{out_dir}/WL_lensit_map_ul.jld2'
phi_file     = f'{out_dir}/phi_maps_lensit_map_ul.jld2'
ckpt_file    = f'{out_dir}/checkpoint_ul.npz'

os.makedirs(out_dir, exist_ok=True)
os.makedirs(iter_base, exist_ok=True)

print(f"\n{'='*60}")
print(f"  UL LensIt MAP")
print(f"  NlevT={NlevT} µK-arcmin  beam={beamFWHM}'  ellmax={ellmax}")
print(f"  N_sims={N_sims}  N_iter={N_iter}  CG_tol={CG_tol}  CG_maxiter={CG_maxiter}")
print(f"{'='*60}\n")

# Cls (from export_julia_cls.jl)
_julia_cls = 'results/lensit/julia_cls.npz'
assert os.path.exists(_julia_cls), f"Missing {_julia_cls} — run julia export_julia_cls.jl"
_d = np.load(_julia_cls)

def _pad(arr, n):
    out = np.zeros(n); out[:min(len(arr), n)] = arr[:min(len(arr), n)]; return out

cl_tt_unl = _pad(_d['cl_tt_unl'], ellmax + 1)
cl_tt_len = _pad(_d['cl_tt_len'], ellmax + 1)
cl_pp     = _pad(_d['cl_pp'],     ellmax + 1)
print(f"Loaded Julia Cls (ell=0..{len(_d['ells'])-1})")
print(f"  cl_pp[100]={cl_pp[100]:.4e}  cl_tt_unl[100]={cl_tt_unl[100]:.4f}  cl_tt_len[100]={cl_tt_len[100]:.4f}")

cls_unl  = {'tt': cl_tt_unl, 'ee': np.zeros(ellmax+1), 'bb': np.zeros(ellmax+1),
            'te': np.zeros(ellmax+1), 'pp': cl_pp}
cls_len  = {'tt': cl_tt_len, 'ee': np.zeros(ellmax+1), 'bb': np.zeros(ellmax+1),
            'te': np.zeros(ellmax+1)}
cl_transf = gauss_beam(beamFWHM / 60. * np.pi / 180., lmax=ellmax)
cls_noise = {'t': (NlevT * np.pi / (180. * 60.))**2 * np.ones(ellmax + 1)}

# LensIt flat-sky patch
print("Building LensIt patch/filter libraries ...")
ellmat_obj = get_ellmat(LD_res, HD_res=HD_res)
lib_alm    = ffs_alm_pyFFTW(ellmat_obj,
               filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax))
isocov     = ffs_cov.ffs_diagcov_alm(
               lib_dir_iso, lib_alm, cls_unl, cls_len, cl_transf, cls_noise)
lib_qlm    = isocov.lib_skyalm
print(f"  lib_qlm.ellmax={lib_qlm.ellmax}  lib_alm shape={lib_alm.shape}")

# N0 curves
print("Computing N0 ...")
N0_unl = isocov.get_N0cls('T', lib_qlm, use_cls_len=False)[0]
N0_len = isocov.get_N0cls('T', lib_qlm, use_cls_len=True)[0]
cpp    = cl_pp[:lib_qlm.ellmax + 1]

def cli(cl):
    r = np.zeros_like(cl, dtype=float)
    r[cl > 0] = 1. / cl[cl > 0]
    return r

_N0 = N0_unl[:lib_qlm.ellmax + 1]
H0_diag = cli(_N0)  # likelihood curvature = 1/N0_unl (standard lensit convention)
print(f"  N0_unl max={N0_unl.max():.4e}  N0_len max={N0_len.max():.4e}  H0_diag max={H0_diag.max():.4e}")

# Simulation libraries
print("Building simulation libraries ...")
lib_lencmb = ffs_cmbs.sims_cmb_len(
    os.path.join(lib_dir_sims, 'lencmbs'), lib_alm, cls_unl, cache_lens=True)
lib_sims   = ffs_maps.lib_noisemap(
    os.path.join(lib_dir_sims, 'maps'),
    lib_alm, lib_lencmb, cl_transf, nTpix, nTpix, nTpix)

# Iterator filter + CG chain
f_id = ffs_deflect.ffs_id_displacement(lib_alm.shape, lib_alm.lsides)
filt_ideal  = ffs_ninv_filt_ideal_nufft.ffs_ninv_filt_wl(
    lib_alm, lib_alm, cls_unl, cl_transf, NlevT, nlev_p=1e6, f=f_id)
chain_descr = chain_samples.get_isomgchain(
    filt_ideal.lib_skyalm.ellmax, filt_ideal.lib_datalm.shape,
    tol=CG_tol, iter_max=CG_maxiter)
opfilt_cinv._type = 'T'

# Spectral binning
edges   = np.arange(ellmin, ellmax + 1, Δℓ)
ell_lo, ell_hi = edges[:-1], edges[1:]
ell_mid = 0.5 * (ell_lo + ell_hi)
nbins   = len(ell_mid)

_cl_norm = (θpix_rad / 512) ** 2
lx    = 2 * np.pi * np.fft.fftfreq(512, d=θpix_rad)
ell2d = np.sqrt(lx[None, :]**2 + lx[:, None]**2)

def safe_div(a, b):
    return np.where(np.abs(b) > 0, a / b, 0.)

def cross_cl_binned(m1, m2):
    cross = np.real(np.fft.fft2(m1) * np.conj(np.fft.fft2(m2))) * _cl_norm
    out   = np.zeros(nbins)
    for i, (lo, hi) in enumerate(zip(ell_lo, ell_hi)):
        msk = (ell2d >= lo) & (ell2d < hi)
        out[i] = cross[msk].mean() if msk.any() else 0.
    return out

# JLD2-compatible HDF5 helpers
_JLD2_MAGIC     = b"HDF5-based Julia Data Format, version 0.1.1"
_JLD2_USERBLOCK = _JLD2_MAGIC + b'\x00' * (512 - len(_JLD2_MAGIC))

def _write_jld2_userblock(fn):
    with open(fn, 'r+b') as fh:
        fh.seek(0); fh.write(_JLD2_USERBLOCK)

def h5_write_sim(filename, sim_key, arrays):
    is_new = not os.path.exists(filename)
    with h5py.File(filename, 'a', **({"userblock_size": 512} if is_new else {})) as f:
        if sim_key in f:
            return
        grp = f.require_group(sim_key)
        for k, v in arrays.items():
            grp.create_dataset(k,
                data=np.int64(v) if isinstance(v, (int, np.integer))
                     else np.asarray(v, dtype=np.float64))
    if is_new:
        _write_jld2_userblock(filename)

def h5_save_wl(filename, data):
    with h5py.File(filename, 'w') as f:
        for k, v in data.items():
            if v is not None:
                f.create_dataset(k, data=np.asarray(v, dtype=np.float64))

# Resume checkpoint
R_mj_sims    = []
sum_R_qe_raw = np.zeros(nbins)
nsims_done   = 0
if os.path.exists(ckpt_file):
    d = np.load(ckpt_file, allow_pickle=True)
    R_mj_sims    = list(d['R_mj_sims'])
    sum_R_qe_raw = d['sum_R_qe_raw']
    nsims_done   = int(d['nsims_done'])
    print(f"Resumed: {nsims_done}/{N_sims} sims done")

# Main loop
for idx in range(nsims_done, N_sims):
    sim_key = f'sim_{idx+1}'
    t_wall = time.strftime('%H:%M:%S')
    print(f"\n{'─'*50}", flush=True)
    print(f"  Sim {idx+1}/{N_sims}  [{t_wall}]", flush=True)

    try:
        plm_true = lib_lencmb.get_sim_plm(idx)
        phi_true = lib_alm.alm2map(plm_true)
        T_obs    = lib_sims.get_sim_tmap(idx)
        datalms  = np.array([lib_alm.map2alm(T_obs)])

        # QE: for T-only, no 0.5 averaging factor (unlike QU which sums two fields).
        # Empirically: 2.0 × N0_unl × get_qlms gives norm_ratio ≈ 1 vs phi_true.
        iblm = isocov.get_iblms('T', datalms, use_cls_len=True)[0]
        plm_qe = 2.0 * isocov.get_qlms('T', iblm, lib_qlm, use_cls_len=False)[0]
        lib_qlm.almxfl(plm_qe, N0_unl[:lib_qlm.ellmax + 1], inplace=True)
        phi_qe = lib_qlm.alm2map(plm_qe)
        # MAP warm start: Wiener-filtered QE (standard lensit convention)
        _wf = cpp * cli(cpp + _N0)
        plm_wf = lib_qlm.almxfl(plm_qe, _wf, inplace=False)

        _norm = lambda p: float(np.sqrt(np.real(np.vdot(p, p))))
        print(f"  QE norm={_norm(plm_qe):.4e}  true norm={_norm(plm_true):.4e}", flush=True)

        # Build iterator (clear stale iter cache each sim)
        iter_dir = os.path.join(iter_base, f'sim{idx:04d}')
        shutil.rmtree(iter_dir, ignore_errors=True)
        os.makedirs(iter_dir)

        itlib = ffs_iterator_pertMF(
            iter_dir, 'T', filt_ideal, datalms, lib_qlm,
            plm_wf, H0_diag, cpp,
            chain_descr=chain_descr, opfilt=opfilt_cinv,
            verbose=False, NR_method=BFGS_MEMORY, nufft_epsilon=1e-7)
        itlib.newton_step_length = lambda _it, ni: (
            min(1.0, 0.2 / ni) if (ni > 0 and np.isfinite(ni)) else 0.5)

        itlib.iterate(0, 'p')
        hist_file = os.path.join(iter_dir, 'history_increment.txt')
        best_grad = 1.0; best_it = 0
        t_sim_start = time.time()
        iter_times  = []   # wall seconds per iteration (for ETA)
        t_last = t_sim_start
        print(f"  it=  0  (QE/WF warm start)", flush=True)

        for it in range(1, N_iter + 1):
            itlib.iterate(it, 'p')

            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            iter_times.append(dt)

            grad_ratio = norm_inc = steplength = None
            if os.path.exists(hist_file):
                try:
                    rows = np.loadtxt(hist_file)
                    if rows.ndim == 1: rows = rows[None, :]
                    if len(rows):
                        row = rows[-1]
                        gr = float(row[3])
                        grad_ratio = gr if np.isfinite(gr) and gr < 1e6 else None
                        norm_inc   = float(row[2]) if len(row) > 2 else None
                        steplength = float(row[7]) if len(row) > 7 else None
                except Exception:
                    pass

            # ETA for this sim and for the full run
            mean_dt = np.mean(iter_times[-10:])   # rolling 10-iter average
            iters_left = N_iter - it
            sims_left  = N_sims - (idx + 1)
            eta_sim_s  = iters_left * mean_dt
            eta_total_s = eta_sim_s + sims_left * N_iter * mean_dt
            eta_sim   = f"{eta_sim_s/60:.0f}m" if eta_sim_s < 3600 else f"{eta_sim_s/3600:.1f}h"
            eta_total = f"{eta_total_s/3600:.1f}h"

            if grad_ratio is not None:
                if grad_ratio < best_grad:
                    best_grad = grad_ratio; best_it = it
                ni_str = f"  step={norm_inc:.4f}" if norm_inc is not None else ""
                sl_str = f"  α={steplength:.4f}"  if steplength is not None else ""
                print(f"  it={it:3d}/{N_iter}  grad={grad_ratio:.5f}  best={best_grad:.5f}@it{best_it}"
                      f"{ni_str}{sl_str}  dt={dt:.0f}s  ETA:{eta_sim}(sim)/{eta_total}(all)", flush=True)
                if best_grad < GRAD_CONV:
                    print(f"  ✓ CONVERGED (grad < {GRAD_CONV})", flush=True)
                    break
                # Early stop if gradient growing after already converging well
                if it >= 4 and grad_ratio > 2.0 * best_grad and best_grad < 1.0:
                    print(f"  ✗ DIVERGING (grad={grad_ratio:.4f} > 2×best={best_grad:.4f}), stopping at best_it={best_it}", flush=True)
                    break
            else:
                print(f"  it={it:3d}/{N_iter}  (spike/nan)  dt={dt:.0f}s  ETA:{eta_sim}(sim)/{eta_total}(all)", flush=True)

        sim_wall = time.time() - t_sim_start
        print(f"  Using it={best_it}  final grad_ratio={best_grad:.5f}  sim_time={sim_wall/60:.1f}min", flush=True)
        phi_map = itlib.get_Phimap(best_it, 'p')

        # Spectra
        cl_tt  = cross_cl_binned(phi_true, phi_true)
        cl_tqe = cross_cl_binned(phi_true, phi_qe)
        cl_tmap= cross_cl_binned(phi_true, phi_map)
        R_mj_sims.append(safe_div(cl_tmap, cl_tt))
        sum_R_qe_raw += safe_div(cl_tqe, cl_tt)
        nsims_done += 1

        # Save phi maps (MAP + QE)
        h5_write_sim(phi_file,    sim_key, {'ϕ_true': phi_true, 'ϕ_mj':     phi_map, 'seed': idx + 1})
        h5_write_sim(qe_phi_file, sim_key, {'ϕ_true': phi_true, 'ϕ_qe_raw': phi_qe,  'seed': idx + 1})

        # Save WL + checkpoint
        W_mj = np.array([np.median([r[i] for r in R_mj_sims]) for i in range(nbins)])
        W_qe = sum_R_qe_raw / nsims_done
        np.savez(ckpt_file, R_mj_sims=np.array(R_mj_sims),
                 sum_R_qe_raw=sum_R_qe_raw, nsims_done=nsims_done)
        h5_save_wl(wl_file, {
            'ℓ_template':      ell_mid,
            'W_mj':            W_mj,
            'W_qe_raw':        W_qe,
            'R_mj_sims':       np.array(R_mj_sims),
            'nsims_map_done':  nsims_done,
            'nsims_completed': nsims_done,
        })
        h5_save_wl(qe_wl_file, {
            'ℓ_template':      ell_mid,
            'W_qe_raw':        W_qe,
            'sum_R_qe_raw':    sum_R_qe_raw,
            'nsims_completed': nsims_done,
        })

        print(f"\n  W_L @ {nsims_done} sim(s):")
        print(f"  {'L':>7}  {'W_QE':>8}  {'W_MAP':>8}")
        for lchk in [500, 1000, 2000, 3000, 5000, 7000, 9000, 11000]:
            i = np.argmin(np.abs(ell_mid - lchk))
            print(f"  {int(ell_mid[i]):>7d}  {W_qe[i]:>8.4f}  {W_mj[i]:>8.4f}")

    except Exception as e:
        import traceback
        print(f"  ✗ SIM {idx+1} CRASHED — skipping\n{traceback.format_exc()}", flush=True)

print(f"\nDone! {nsims_done}/{N_sims} sims.")
print(f"  WL:  {wl_file}")
print(f"  phi: {phi_file}")

