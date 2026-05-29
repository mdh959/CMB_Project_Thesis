#!/usr/bin/env julia
# fig1_reconstruction_maps.jl
# Five-panel figure: κ_true, κ_GI, |∇T_LP|, κ_QE, κ_MAP
# Layout: top row (a–c), bottom row (d–e) centred.
# Runs S4 and UL cases, saves two figures.

import Pkg; Pkg.activate(@__DIR__)
using CMBLensing, Statistics, JLD2, PythonPlot, PythonCall, Printf, LinearAlgebra
import CMBLensing: m_rfft, m_irfft

# Shared geometry
const θpix       = 0.7438046267475303
const θpix_rad   = θpix * π / (180 * 60)
const Nside      = 512
const Lhp        = 4000
const Lmax       = 12000
const Lgrad      = 2000
const RAD2AMIN   = 60 * 180 / π
const Np         = 138
const TARGET_SEED = 1001
const OUT_DIR    = "results"

# Helpers
function fftfreq_ints(N)
    iseven(N) ? vcat(0:(N÷2), (-(N÷2-1)):-1) : vcat(0:((N-1)÷2), (-((N-1)÷2)):-1)
end

function make_ellxy_grid_rfft(N, tpr)
    NyF  = N÷2 + 1
    Lbox = N * tpr
    ℓy   = 2π .* collect(0:(NyF-1)) ./ Lbox
    ℓx   = 2π .* fftfreq_ints(N)    ./ Lbox
    return repeat(ℓx', NyF, 1), repeat(ℓy, 1, N),
           sqrt.(repeat(ℓx', NyF, 1).^2 .+ repeat(ℓy, 1, N).^2)
end

function smooth_1d(W; window=9)
    Ws = similar(W); hw = window ÷ 2
    for i in eachindex(W)
        v = filter(isfinite, W[max(firstindex(W), i-hw):min(lastindex(W), i+hw)])
        Ws[i] = isempty(v) ? 0.0 : mean(v)
    end
    return Ws
end

function wiener_kappa(phi_arr::Matrix{Float64}, ℓ2d::Matrix{Float64},
                       Cpp::Cℓs, N0::Cℓs;
                       Llo=Float64(Lhp), Lhi=Float64(Lmax),
                       WL::Union{Cℓs,Nothing}=nothing, minW=1e-8)
    N  = size(phi_arr, 1)
    F  = m_rfft(phi_arr, (1,2))
    cL_lo = minimum(Cpp.ℓ); cL_hi = maximum(Cpp.ℓ)
    nL_lo = minimum(N0.ℓ);  nL_hi = maximum(N0.ℓ)
    wL_lo = WL === nothing ? 0.0 : minimum(WL.ℓ)
    wL_hi = WL === nothing ? 0.0 : maximum(WL.ℓ)
    for j in axes(F,2), i in axes(F,1)
        L = ℓ2d[i,j]
        if L < Llo || L > Lhi
            F[i,j] = 0.0+0.0im; continue
        end
        c  = max(Float64(Cpp(clamp(L, cL_lo, cL_hi))), 0.0)
        n  = max(Float64(N0(clamp(L,  nL_lo, nL_hi))), 1e-60)
        if WL !== nothing
            # Biased estimator: ϕ_est ≈ WL·ϕ_true + noise.
            # Optimal Wiener filter: C·WL / (WL²·C + N0), no separate /WL step.
            w  = Float64(WL(clamp(L, wL_lo, wL_hi)))
            wt = abs(w) > minW ? (c * w / (w^2 * c + n)) * 0.5 * L^2 : 0.0
        else
            wt = (c > 0 ? c / (c + n) : 0.0) * 0.5 * L^2
        end
        F[i,j] *= wt
    end
    return real.(m_irfft(F, N, (1,2)))
end

# Bandpass κ: ϕ_true has no noise, just apply L²/2 in [Lhp, Lmax].
function bandpass_kappa(phi_arr::Matrix{Float64}, ℓ2d::Matrix{Float64};
                         Llo=Float64(Lhp), Lhi=Float64(Lmax))
    N = size(phi_arr, 1)
    F = m_rfft(phi_arr, (1,2))
    for j in axes(F,2), i in axes(F,1)
        L = ℓ2d[i,j]
        F[i,j] = (L >= Llo && L <= Lhi) ? F[i,j] * 0.5 * L^2 : 0.0+0.0im
    end
    return real.(m_irfft(F, N, (1,2)))
end

function grad_mag(d_field, ℓx2D, ℓy2D)
    arr = Float64.(Map(LowPass(Lgrad) * d_field).arr)
    N   = size(arr, 1)
    F   = m_rfft(arr, (1,2))
    gx  = real.(m_irfft((im .* ℓx2D) .* F, N, (1,2)))
    gy  = real.(m_irfft((im .* ℓy2D) .* F, N, (1,2)))
    return sqrt.(gx.^2 .+ gy.^2)
end

function best_grad_patch(gmag, Np; stride=4)
    N = size(gmag, 1)
    best = (score=-Inf, r0=1, c0=1)
    for r0 in 1:stride:(N-Np+1), c0 in 1:stride:(N-Np+1)
        s = mean(gmag[r0:r0+Np-1, c0:c0+Np-1])
        s > best.score && (best = (score=s, r0=r0, c0=c0))
    end
    return best.r0:best.r0+Np-1, best.c0:best.c0+Np-1
end

# Precompute once
println("Precomputing CAMB spectrum and Fourier grid...")
const Cℓ_cam             = camb(r=0.05, ℓmax=35000)
const ℓx2D, ℓy2D, ℓ2d  = make_ellxy_grid_rfft(Nside, θpix_rad)

# Figure factory
function make_fig1(qegi_file, wl_qegi_file, map_file, wl_map_file,
                   μKarcminT::Float64, beamFWHM::Float64, out_file;
                   seed::Int=TARGET_SEED)
    println("\n── $(basename(out_file)) ──")
    isfile(qegi_file)    || (println("  SKIP: $qegi_file not found");    return)
    isfile(map_file)     || (println("  SKIP: $map_file not found");     return)
    isfile(wl_qegi_file) || (println("  SKIP: $wl_qegi_file not found"); return)
    isfile(wl_map_file)  || (println("  SKIP: $wl_map_file not found");  return)

    # ── Load WL ───────────────────────────────────────────────────────────────
    println("  Loading WL files...")
    d_wl_qegi = JLD2.load(wl_qegi_file)
    d_wl_map  = JLD2.load(wl_map_file)
    ell_gi    = Float64.(d_wl_qegi["ℓ_template"])
    ell_map   = Float64.(d_wl_map["ℓ_template"])
    _lw(d, k) = (haskey(d, k) && d[k] !== nothing) ? Float64.(d[k]) : nothing
    W_qe_raw  = something(_lw(d_wl_qegi, "W_qe_raw"), _lw(d_wl_qegi, "W_qe_wf"))
    W_qe_raw === nothing && error("W_qe_raw/W_qe_wf missing from $wl_qegi_file")
    WL_gi  = Cℓs(ell_gi,  smooth_1d(Float64.(d_wl_qegi["W_gi_b"])))
    WL_qe  = Cℓs(ell_gi,  smooth_1d(W_qe_raw))
    WL_map = Cℓs(ell_map, smooth_1d(Float64.(d_wl_map["W_mj"])))

    # ── Load ϕ arrays ─────────────────────────────────────────────────────────
    println("  Loading ϕ arrays (seed=$seed)...")
    ϕ_true_arr = ϕ_gi_arr = ϕ_qe_arr = ϕ_map_arr = nothing
    jldopen(qegi_file, "r") do f
        for key in sort(collect(keys(f)))
            match(r"^sim_\d+$", key) === nothing && continue
            s = read(f, "$key/seed")
            (s isa AbstractArray ? Int(first(s)) : Int(s)) == seed || continue
            ϕ_true_arr = Float64.(read(f, "$key/ϕ_true"))
            ϕ_gi_arr   = Float64.(read(f, "$key/ϕ_gi_b"))
            ϕ_qe_arr   = Float64.(read(f, "$key/ϕ_qe_raw"))
            break
        end
    end
    jldopen(map_file, "r") do f
        for key in sort(collect(keys(f)))
            match(r"^sim_\d+$", key) === nothing && continue
            s = read(f, "$key/seed")
            (s isa AbstractArray ? Int(first(s)) : Int(s)) == seed || continue
            ϕ_map_arr = Float64.(read(f, "$key/ϕ_mj"))
            break
        end
    end
    ϕ_true_arr === nothing && error("seed $seed not found in $qegi_file")
    ϕ_map_arr  === nothing && error("seed $seed not found in $map_file")

    # ── Reload simulation ─────────────────────────────────────────────────────
    println("  Reloading simulation (μK=$μKarcminT, beam=$beamFWHM)...")
    Cℓn = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
    (; ϕ, ds) = load_sim(seed=seed, Cℓ=Cℓ_cam, Cℓn=Cℓn,
                          θpix=θpix, T=Float64, Nside=Nside,
                          beamFWHM=beamFWHM, pol=:I,
                          bandpass_mask=LowPass(Lmax),
                          pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0))

    wrap(arr)  = typeof(Map(ϕ))(Float64.(arr), ϕ.metadata)
    Cpp_spec   = get_Cℓ(wrap(ϕ_true_arr); Δℓ=100)

    # ── Empirical N0 for all: power of (ϕ_est − W_L·ϕ_true) ─────────────────
    println("  Computing empirical N0 for GI, QE, MAP (up to 50 sims each)...")
    wL_lo_gi = minimum(WL_gi.ℓ);  wL_hi_gi = maximum(WL_gi.ℓ)
    wL_lo_qe = minimum(WL_qe.ℓ);  wL_hi_qe = maximum(WL_qe.ℓ)
    wL_lo_m  = minimum(WL_map.ℓ); wL_hi_m  = maximum(WL_map.ℓ)
    n0_gi_sum = nothing; n0_gi_cnt = 0
    n0_qe_sum = nothing; n0_qe_cnt = 0
    jldopen(qegi_file, "r") do f
        for key in sort(collect(keys(f)))
            match(r"^sim_\d+$", key) === nothing && continue
            haskey(f[key], "ϕ_gi_b") && haskey(f[key], "ϕ_qe_raw") && haskey(f[key], "ϕ_true") || continue
            phi_tr = Float64.(read(f, "$key/ϕ_true"))
            phi_gi = Float64.(read(f, "$key/ϕ_gi_b"))
            phi_qe = Float64.(read(f, "$key/ϕ_qe_raw"))
            N      = size(phi_tr, 1)
            Ftr    = m_rfft(phi_tr, (1,2))
            Fgi    = copy(Ftr); Fqe = copy(Ftr)
            for jj in axes(Ftr,2), ii in axes(Ftr,1)
                L = ℓ2d[ii,jj]
                Fgi[ii,jj] *= Float64(WL_gi(clamp(L, wL_lo_gi, wL_hi_gi)))
                Fqe[ii,jj] *= Float64(WL_qe(clamp(L, wL_lo_qe, wL_hi_qe)))
            end
            r_gi = Float64.(get_Cℓ(wrap(phi_gi .- real.(m_irfft(Fgi, N, (1,2)))); Δℓ=100).Cℓ)
            r_qe = Float64.(get_Cℓ(wrap(phi_qe .- real.(m_irfft(Fqe, N, (1,2)))); Δℓ=100).Cℓ)
            n0_gi_sum = n0_gi_sum === nothing ? copy(r_gi) : n0_gi_sum .+ r_gi
            n0_qe_sum = n0_qe_sum === nothing ? copy(r_qe) : n0_qe_sum .+ r_qe
            n0_gi_cnt += 1; n0_qe_cnt += 1
            n0_gi_cnt >= 50 && break
        end
    end
    n0_map_sum = nothing; n0_map_cnt = 0
    jldopen(map_file, "r") do f
        for key in sort(collect(keys(f)))
            match(r"^sim_\d+$", key) === nothing && continue
            haskey(f[key], "ϕ_mj") && haskey(f[key], "ϕ_true") || continue
            phi_mj = Float64.(read(f, "$key/ϕ_mj"))
            phi_tr = Float64.(read(f, "$key/ϕ_true"))
            N      = size(phi_tr, 1)
            Ftr    = m_rfft(phi_tr, (1,2))
            for jj in axes(Ftr,2), ii in axes(Ftr,1)
                Ftr[ii,jj] *= Float64(WL_map(clamp(ℓ2d[ii,jj], wL_lo_m, wL_hi_m)))
            end
            r_m = Float64.(get_Cℓ(wrap(phi_mj .- real.(m_irfft(Ftr, N, (1,2)))); Δℓ=100).Cℓ)
            n0_map_sum = n0_map_sum === nothing ? copy(r_m) : n0_map_sum .+ r_m
            n0_map_cnt += 1
            n0_map_cnt >= 50 && break
        end
    end
    N0_gi  = Cℓs(Cpp_spec.ℓ, smooth_1d(max.(n0_gi_sum  ./ n0_gi_cnt,  1e-60)))
    N0_qe  = Cℓs(Cpp_spec.ℓ, smooth_1d(max.(n0_qe_sum  ./ n0_qe_cnt,  1e-60)))
    N0_map = Cℓs(Cpp_spec.ℓ, smooth_1d(max.(n0_map_sum ./ n0_map_cnt, 1e-60)))
    println("  N0_gi: $n0_gi_cnt sims, N0_qe: $n0_qe_cnt sims, N0_map: $n0_map_cnt sims")

    # ── Wiener-filter and gradient ────────────────────────────────────────────
    println("  Wiener-filtering all maps...")
    κ_true = bandpass_kappa(ϕ_true_arr, ℓ2d)
    κ_gi   = wiener_kappa(ϕ_gi_arr,   ℓ2d, Cpp_spec, N0_gi;  WL=WL_gi)
    κ_qe   = wiener_kappa(ϕ_qe_arr,   ℓ2d, Cpp_spec, N0_qe;  WL=WL_qe)
    κ_map  = wiener_kappa(ϕ_map_arr,  ℓ2d, Cpp_spec, N0_map; WL=WL_map)
    gmag   = grad_mag(ds.d, ℓx2D, ℓy2D)

    # ── Choose patch ──────────────────────────────────────────────────────────
    rows, cols = best_grad_patch(gmag, Np)
    patch_deg  = Np * θpix / 60
    r0_d = (first(rows)-1) * θpix / 60;  r1_d = last(rows)  * θpix / 60
    c0_d = (first(cols)-1) * θpix / 60;  c1_d = last(cols)  * θpix / 60
    ext  = [c0_d, c1_d, r0_d, r1_d]

    κt_p = κ_true[rows, cols];  κg_p = κ_gi[rows, cols]
    κq_p = κ_qe[rows,  cols];   κm_p = κ_map[rows, cols]
    gm_p = gmag[rows, cols] ./ RAD2AMIN

    ρ_gi  = cor(vec(κt_p), vec(κg_p))
    ρ_qe  = cor(vec(κt_p), vec(κq_p))
    ρ_map = cor(vec(κt_p), vec(κm_p))
    @printf("  Patch %.2f×%.2f deg | ρ_GI=%.3f  ρ_QE=%.3f  ρ_MAP=%.3f\n",
            patch_deg, patch_deg, ρ_gi, ρ_qe, ρ_map)

    # ── Figure ────────────────────────────────────────────────────────────────
    PythonPlot.rc("font",     family="serif", size=11)
    PythonPlot.rc("axes",     linewidth=0.6)
    PythonPlot.rc("xtick",    direction="in", top=true,   labelsize=9)
    PythonPlot.rc("ytick",    direction="in", right=true, labelsize=9)
    PythonPlot.rc("mathtext", fontset="cm")

    # Layout geometry
    _L=0.07; _R=0.87; _T=0.95; _B=0.16; _hs=0.18; _ws=0.06
    _pw = (_R-_L) / (3 + 2*_ws)
    _ph = (_T-_B) / (2 +   _hs)
    _wg = _ws*_pw; _hg = _hs*_ph
    _ry = [_T - k*_ph - (k-1)*_hg for k in 1:2]

    fig, axs = PythonPlot.subplots(2, 3; figsize=(12.0, 9.0))
    fig.subplots_adjust(hspace=_hs, wspace=_ws, left=_L, right=_R, top=_T, bottom=_B)
    axs[1, 2].set_visible(false)

    # Centre bottom row
    _bx0 = (_L + _R) / 2 - _pw - _wg / 2
    axs[1, 0].set_position([_bx0,          _ry[2], _pw, _ph])
    axs[1, 1].set_position([_bx0 + _pw + _wg, _ry[2], _pw, _ph])

    vm_κ   = max(quantile(abs.(vcat(vec(κt_p), vec(κg_p), vec(κq_p), vec(κm_p))), 0.97), 1e-8)
    vm_g   = quantile(vec(gm_p), 0.99)
    cmap_κ = "RdBu_r"
    cmap_g = "inferno"
    ik = Dict(:origin=>"lower", :extent=>ext, :aspect=>"equal", :interpolation=>"nearest")

    plabel(ax, lbl) = ax.text(-0.04, 1.02, lbl; transform=ax.transAxes, fontsize=12,
                              fontweight="bold", va="bottom", ha="right", clip_on=false)

    im0 = axs[0,0].imshow(κt_p; cmap=cmap_κ, vmin=-vm_κ, vmax=vm_κ, ik...)
    plabel(axs[0,0], "(a)")
    axs[0,0].set_title(L"Input $\kappa_{\rm true}$"; fontsize=11, pad=3)
    axs[0,0].set_ylabel("Dec (deg)"; fontsize=11)
    axs[0,0].tick_params(labelbottom=false)

    axs[0,1].imshow(κg_p; cmap=cmap_κ, vmin=-vm_κ, vmax=vm_κ, ik...)
    plabel(axs[0,1], "(b)")
    axs[0,1].set_title("GI reconstruction"; fontsize=11, pad=3)
    axs[0,1].tick_params(labelbottom=false, labelleft=false)

    im2 = axs[0,2].imshow(gm_p; cmap=cmap_g, vmin=0, vmax=vm_g, ik...)
    plabel(axs[0,2], "(c)")
    axs[0,2].set_title("Low-pass CMB gradient magnitude"; fontsize=11, pad=3)
    axs[0,2].tick_params(labelbottom=false, labelleft=false)

    axs[1,0].imshow(κq_p; cmap=cmap_κ, vmin=-vm_κ, vmax=vm_κ, ik...)
    plabel(axs[1,0], "(d)")
    axs[1,0].set_title("QE reconstruction"; fontsize=11, pad=3)
    axs[1,0].set_ylabel("Dec (deg)"; fontsize=11)
    axs[1,0].set_xlabel("RA (deg)"; fontsize=11)

    axs[1,1].imshow(κm_p; cmap=cmap_κ, vmin=-vm_κ, vmax=vm_κ, ik...)
    plabel(axs[1,1], "(e)")
    axs[1,1].set_title("joint-MAP reconstruction"; fontsize=11, pad=3)
    axs[1,1].set_xlabel("RA (deg)"; fontsize=11)
    axs[1,1].tick_params(labelleft=false)

    # ── Colorbars ─────────────────────────────────────────────────────────────
    _cbw = 0.018; _gap = 0.012
    pos_c = axs[0,2].get_position()
    pos_d = axs[1,0].get_position()
    pos_e = axs[1,1].get_position()

    cax_g = fig.add_axes([pyconvert(Float64,pos_c.x1)+_gap, pyconvert(Float64,pos_c.y0),
                           _cbw, pyconvert(Float64,pos_c.height)])
    cb_g  = fig.colorbar(im2; cax=cax_g)
    cb_g.set_label(L"$|\nabla T_{\rm lp}|\,[\mu{\rm K\,arcmin}^{-1}]$"; fontsize=11, labelpad=4)
    cb_g.ax.tick_params(labelsize=11)

    _cbh = 0.012
    _cby = pyconvert(Float64, pos_d.y0) - 0.068
    cax_κ = fig.add_axes([pyconvert(Float64, pos_d.x0), _cby,
                           pyconvert(Float64, pos_e.x1) - pyconvert(Float64, pos_d.x0), _cbh])
    cb_κ  = fig.colorbar(im0; cax=cax_κ, orientation="horizontal")
    cb_κ.set_label(L"$\kappa$, filtered to $4000 < L < 12000$"; fontsize=11, labelpad=3)
    cb_κ.set_ticks([-vm_κ, 0.0, vm_κ])
    cb_κ.ax.set_xticklabels([@sprintf("%.2g", -vm_κ), "0", @sprintf("%.2g", vm_κ)])
    cb_κ.ax.tick_params(labelsize=11)

    # ── Save ──────────────────────────────────────────────────────────────────
    mkpath(OUT_DIR)
    fig.savefig(out_file; dpi=200)
    println("  Saved → $out_file")
    fig.savefig(replace(out_file, ".pdf" => ".png"); dpi=200, bbox_inches="tight")
    println("  Saved → $(replace(out_file, ".pdf" => ".png"))")
    PythonPlot.plotclose("all")
end

# S4 case (1 µK-arcmin, 1' beam)
make_fig1(
    "results/phi_maps_qe_gi_12000.jld2",
    "results/WL_qe_gi_12000.jld2",
    "results/phi_maps_map_12000.jld2",
    "results/WL_map_12000.jld2",
    1.0, 1.0,
    joinpath(OUT_DIR, "fig1_gi_qe_map_gradient_s4.pdf"))

# UL case (0.1 µK-arcmin, 0.3' beam)
let
    ul_wl_map  = isfile("results/WL_map_12000_ul_zero_a01.jld2")       ? "results/WL_map_12000_ul_zero_a01.jld2"       :
                 isfile("results/WL_map_12000_ul.jld2")                ? "results/WL_map_12000_ul.jld2"                : ""
    ul_phi_map = isfile("results/phi_maps_map_12000_ul_zero_a01.jld2") ? "results/phi_maps_map_12000_ul_zero_a01.jld2" :
                 isfile("results/phi_maps_map_12000_ul.jld2")          ? "results/phi_maps_map_12000_ul.jld2"          : ""
    make_fig1(
        "results/phi_maps_qe_gi_12000_ul.jld2",
        "results/WL_qe_gi_12000_ul.jld2",
        ul_phi_map, ul_wl_map,
        0.1, 0.3,
        joinpath(OUT_DIR, "fig1_gi_qe_map_gradient_ul.pdf"))
end

println("\nDone.")
