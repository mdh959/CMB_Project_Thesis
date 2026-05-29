#!/usr/bin/env julia
# run_qe_gi_wl12k.jl
#
# QE / GI / MAP joint pipeline.  Output files (per noise level):
#   phi_maps_qe_gi_12000{suffix}.jld2  — ϕ_true, ϕ_qe_raw, ϕ_gi_b, N0_* per sim
#   WL_qe_gi_12000{suffix}.jld2        — W_L transfer functions
#   phi_maps_map_12000{suffix}.jld2    — ϕ_true, ϕ_mj per MAP sim
#   WL_map_12000{suffix}.jld2          — MAP W_L, logpdf histories
#
#   S4  (1 µK-arcmin): QE-WF warm start, αmax=0.3,  50 steps, CG nsteps=200
#   UL (0.1 µK-arcmin): QE/GI hybrid warm start, αmax=0.05, 50 steps, CG nsteps=500

import Pkg; Pkg.activate(@__DIR__)
using CMBLensing, LinearAlgebra
import CMBLensing.Setfield: @set
using Statistics: mean, median
using JLD2, Printf
include("utils.jl"); using .Utils

# Patch CMBLensing preconditioner: use lensed Cf̃ instead of unlensed Cf.
# Cf̃ better approximates the MAP Hessian, reducing CG iterations at low noise.
@eval CMBLensing begin
    function Hessian_logpdf_preconditioner(::Val{:f}, ds::DataSet)
        @unpack Cf̃, B̂, M̂, Cn̂ = ds
        v   = copy(diag(pinv(Cf̃) + B̂'*M̂'*pinv(Cn̂)*M̂*B̂))
        arr = v.arr
        good = isfinite.(arr) .& (arr .> 0)
        fill!(view(arr, .!good), any(good) ? mean(arr[good]) : 1.0)
        arr .= clamp.(arr, 1e-30, 1e30)
        return Diagonal(v)
    end
end

const Cℓ         = camb(r=0.05, ℓmax=35000)
const θpix       = 0.7438046267475303    # pixel width (arcmin); ℓ_Nyquist ≈ 14500
const Nside      = 512
const pol        = :I
const nsims      = 100
const seed0      = 1000
const Δℓ         = 30
const nsims_map  = 500
const MAPJ_STEPS = 40

println("f_sky_patch = $(round((Nside*θpix*π/(180*60))^2/(4π); sigdigits=4))")

function run_noise_level(μKarcminT::Float64, suffix::String, Lmax::Int, beamFWHM::Float64;
                         Δℓ_wl::Int=Δℓ, run_map::Bool=true, run_rdn0::Bool=false,
                         run_convergence_diag::Bool=false,
                         θpix_sim::Float64=θpix, Nside_sim::Int=Nside,
                         map_Lmax::Int=Lmax, map_αmax::Float64=0.05,
                         map_zero_start::Bool=false, map_warmstart_Lmax::Int=0,
                         map_file_suffix::String=suffix,
                         map_cg_tolerance::Float64=1e-4,
                         map_cg_nsteps::Int=500,
                         map_nburnin_hessian::Int=typemax(Int),
                         map_prior_deproj::Float64=0.0,
                         map_prior_weakening::Float64=1.0,
                         map_hl_prior_weakening::Float64=1.0,
                         map_hl_prior_Lmin::Int=3000,
                         map_nsims::Int=nsims_map,
                         map_nsteps::Int=MAPJ_STEPS,
                         map_warmstart_weights::Symbol=:unlensed,
                         map_warmstart_scale::Float64=1.0,
                         map_extra_exclude::Set{Int}=Set{Int}(),
                         qe_Lmax::Int=Lmax)

    println("\n" * "="^70)
    println("=== μKarcminT=$μKarcminT, Lmax=$Lmax, beam=$(beamFWHM)′  (suffix=\"$suffix\") ===")
    println("    θpix=$(θpix_sim)′, Nside=$(Nside_sim), ℓ_Nyquist≈$(round(Int, π/(θpix_sim*π/(180*60))))")
    println("="^70)

    Cℓn         = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
    # Hard cutoff at qe_Lmax — no taper. Taper caused 1/ΣTtot blow-up in the
    # transition zone (ΣTtot→0 while data doesn't track it), producing stripe outliers.
    # For 0.3′ beam B(L≈11000)≈2e-4 so near-Lmax modes carry negligible lensing SNR.
    qe_bandpass = LowPass(qe_Lmax)

    WL_file       = "results/WL_qe_gi_12000$(suffix).jld2"
    phi_maps_file = "results/phi_maps_qe_gi_12000$(suffix).jld2"

    load_kwargs = (
        Cℓ=Cℓ, Cℓn=Cℓn, θpix=θpix_sim, T=Float64, Nside=Nside_sim,
        beamFWHM=beamFWHM, pol=pol, bandpass_mask=LowPass(Lmax),
        pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
    )

    safe_div(a, b) = @. ifelse(abs(b) > 0.0, a / b, 0.0)

    # Resume checkpoint
    sum_R_qe_raw = sum_R_gi_b = W_qe_raw = W_gi_b = ℓ_template = nothing
    nsims_completed = 0; seeds_done = Int[]
    if isfile(WL_file)
        d = JLD2.load(WL_file)
        lf(k) = (haskey(d,k) && d[k] !== nothing) ? Float64.(d[k]) : nothing
        sum_R_qe_raw = lf("sum_R_qe_raw"); sum_R_gi_b = lf("sum_R_gi_b")
        W_qe_raw = something(lf("W_qe_raw"), lf("W_qe_wf"))   # backward compat
        W_gi_b   = lf("W_gi_b");  ℓ_template = lf("ℓ_template")
        haskey(d,"nsims_completed") && (nsims_completed = d["nsims_completed"])
        haskey(d,"seeds_done")      && (seeds_done      = d["seeds_done"])
        println("Resumed W_L checkpoint: $nsims_completed / $nsims sims")
    end

    phi_sims_done = Set{Int}()
    if isfile(phi_maps_file)
        try
            jldopen(phi_maps_file, "r") do f
                for key in keys(f)
                    m = match(r"^sim_(\d+)$", key)
                    m !== nothing && push!(phi_sims_done, parse(Int, m.captures[1]))
                end
            end
            println("$(length(phi_sims_done)) phi maps already in $phi_maps_file")
        catch err
            @warn "phi_maps_file corrupted ($err) — deleting"; rm(phi_maps_file)
        end
    end

    # Fiducial GI denominator
    # σ_fid is shared by gi_estimate AND gi_n0_rdn0 so both use identical mode
    # masks. Without this, RDN0 debiases a different estimator than was run.
    σ_fid_gi = if isfile(WL_file) && haskey(JLD2.load(WL_file), "σ_fid_gi_xx")
        d_wl = JLD2.load(WL_file)
        (σ_xx=d_wl["σ_fid_gi_xx"], σ_xy=d_wl["σ_fid_gi_xy"], σ_yy=d_wl["σ_fid_gi_yy"])
    else
        println("Computing σ_fid_gi (nmc=40)..."); flush(stdout)
        sf = compute_sigma_fid(load_sim(; seed=seed0+1, load_kwargs...).ds;
                               nmc=40, Lgrad=2000, seed_start=70_000)
        jldopen(WL_file, "a+") do f
            f["σ_fid_gi_xx"] = sf.σ_xx; f["σ_fid_gi_xy"] = sf.σ_xy; f["σ_fid_gi_yy"] = sf.σ_yy
        end
        println("  σ_xx=$(round(sf.σ_xx,sigdigits=4))  σ_yy=$(round(sf.σ_yy,sigdigits=4))")
        sf
    end

    # QE / GI sim loop
    for s in (nsims_completed + 1):nsims
        seed = seed0 + s
        print("  Sim $s/$nsims (seed=$seed) ... "); flush(stdout)

        (; ϕ, ds) = load_sim(; seed=seed, load_kwargs...)
        qe_Cℓn = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=qe_Lmax)
        ds_qe  = load_sim(; seed=seed, load_kwargs..., bandpass_mask=qe_bandpass, Cℓn=qe_Cℓn).ds

        ϕqe_raw = quadratic_estimate(ds_qe; weights=:unlensed, wiener_filtered=false).ϕqe
        ϕgi_b   = gi_estimate(ds; Lhp=4000, Lmax=Lmax)

        cl_tt    = get_Cℓ(ϕ;           Δℓ=Δℓ_wl)
        cl_tqraw = get_Cℓ(ϕ, ϕqe_raw; Δℓ=Δℓ_wl)
        cl_tgib  = get_Cℓ(ϕ, ϕgi_b;   Δℓ=Δℓ_wl)

        N0_rdn0_Cℓ = run_rdn0 ?
            Float64.(cov_to_Cℓ(N0_bias(ds_qe; weights=:unlensed, realization_spec=:data).N0; Δℓ=Δℓ).Cℓ) : nothing

        gi_n0_fgmc  = gi_n0_fixed_gradient_mc(ds; nmc=20, Lhp=4000, Lmax=Lmax, Δℓ=Δℓ,
                                               seed_start=42_000 + s * 100)
        gi_n0_linrd = gi_n0_linrd_analytical(ds;
                nmc=20, Lgrad=2000, Lhp=4000, Lmax=Lmax, Δℓ=Δℓ,
                seed_start=95_000 + s * 1000,
                smooth_C_data=true, smooth_window=9, clamp_negative=false)
        gi_n0_rd    = gi_n0_rdn0(ds; nsim=20, σ_fid=σ_fid_gi,
                Lhp=4000, Lmax=Lmax, Δℓ=Δℓ, seed_start=70_000 + s * 1000)

        if ℓ_template === nothing
            ℓ_template   = Float64.(collect(cl_tt.ℓ))
            sum_R_qe_raw = zeros(Float64, length(ℓ_template))
            sum_R_gi_b   = zeros(Float64, length(ℓ_template))
        end

        ctt = Float64.(cl_tt.Cℓ)
        sum_R_qe_raw .+= safe_div(Float64.(cl_tqraw.Cℓ), ctt)
        sum_R_gi_b   .+= safe_div(Float64.(cl_tgib.Cℓ),  ctt)
        nsims_completed += 1; push!(seeds_done, seed)
        W_qe_raw = sum_R_qe_raw ./ nsims_completed
        W_gi_b   = sum_R_gi_b   ./ nsims_completed

        jldopen(phi_maps_file, "a+") do f
            if s ∉ phi_sims_done
                f["sim_$s/ϕ_true"]   = Float64.(Map(ϕ).arr)
                f["sim_$s/ϕ_qe_raw"] = Float64.(Map(ϕqe_raw).arr)
                f["sim_$s/ϕ_gi_b"]   = Float64.(Map(ϕgi_b).arr)
                f["sim_$s/seed"]     = seed
                push!(phi_sims_done, s)
            end
            # QE RDN0 (CMBLensing built-in)
            run_rdn0 && N0_rdn0_Cℓ !== nothing && !haskey(f,"sim_$s/N0_rdn0") &&
                (f["sim_$s/N0_rdn0"] = N0_rdn0_Cℓ)
            # GI fg-MC N0 — write if missing or no method tag (pre-v2)
            if !haskey(f,"sim_$s/N0_gi_fgmc_method")
                for k in ("N0_gi_fgmc_ell","N0_gi_fgmc","N0_gi_fgmc_Lhp","N0_gi_fgmc_Lmax","N0_gi_fgmc_method")
                    haskey(f,"sim_$s/$k") && delete!(f,"sim_$s/$k")
                end
                f["sim_$s/N0_gi_fgmc_ell"]    = Float64.(gi_n0_fgmc.ℓ)
                f["sim_$s/N0_gi_fgmc"]         = Float64.(gi_n0_fgmc.Cℓ)
                f["sim_$s/N0_gi_fgmc_Lhp"]    = gi_n0_fgmc.Lhp
                f["sim_$s/N0_gi_fgmc_Lmax"]   = gi_n0_fgmc.Lmax
                f["sim_$s/N0_gi_fgmc_method"]  = gi_n0_fgmc.method
            end
            !haskey(f,"sim_$s/N0_gi_linrd") &&
                (f["sim_$s/N0_gi_linrd_ell"] = Float64.(gi_n0_linrd.ℓ);
                 f["sim_$s/N0_gi_linrd"]     = Float64.(gi_n0_linrd.Cℓ))
            # GI RDN0 (realisation-dependent, uses σ_fid)
            !haskey(f,"sim_$s/N0_gi_rdn0_v2") &&
                (f["sim_$s/N0_gi_rdn0_v2_ell"] = Float64.(gi_n0_rd.ℓ);
                 f["sim_$s/N0_gi_rdn0_v2"]     = Float64.(gi_n0_rd.Cℓ))
        end

        @save WL_file sum_R_qe_raw sum_R_gi_b ℓ_template W_qe_raw W_gi_b nsims_completed seeds_done
        println("done"); flush(stdout)
    end

    # Retroactive fg-MC N0 (safety net + regen old entries without metadata)
    if isfile(phi_maps_file)
        sims_need_fgmc = Int[]
        jldopen(phi_maps_file, "r") do f
            for s in sort(collect(phi_sims_done))
                # regen if missing entirely or no method tag (pre-v2)
                (!haskey(f,"sim_$s/N0_gi_fgmc") || !haskey(f,"sim_$s/N0_gi_fgmc_method")) &&
                    push!(sims_need_fgmc, s)
            end
        end
        if !isempty(sims_need_fgmc)
            println("\nRetroactive fg-MC N0 (v2): $(length(sims_need_fgmc)) sims...")
            jldopen(phi_maps_file, "a+") do f
                for (i, s) in enumerate(sims_need_fgmc)
                    print("\r  fg-MC N0 $i/$(length(sims_need_fgmc))"); flush(stdout)
                    seed_s  = haskey(f,"sim_$s/seed") ? read(f,"sim_$s/seed") : seed0+s
                    sim_idx = haskey(f,"sim_$s/seed") ? (seed_s - seed0) : s
                    fgmc = gi_n0_fixed_gradient_mc(load_sim(; seed=seed_s, load_kwargs...).ds;
                               nmc=20, Lhp=4000, Lmax=Lmax, Δℓ=Δℓ,
                               seed_start=42_000 + sim_idx * 100)
                    for k in ("N0_gi_fgmc_ell","N0_gi_fgmc","N0_gi_fgmc_Lhp","N0_gi_fgmc_Lmax","N0_gi_fgmc_method")
                        haskey(f,"sim_$s/$k") && delete!(f,"sim_$s/$k")
                    end
                    f["sim_$s/N0_gi_fgmc_ell"]   = Float64.(fgmc.ℓ)
                    f["sim_$s/N0_gi_fgmc"]        = Float64.(fgmc.Cℓ)
                    f["sim_$s/N0_gi_fgmc_Lhp"]   = fgmc.Lhp
                    f["sim_$s/N0_gi_fgmc_Lmax"]  = fgmc.Lmax
                    f["sim_$s/N0_gi_fgmc_method"] = fgmc.method
                end
                println()
            end
            println("  Done.")
        end
    end

    # Retroactive QE RDN0
    if run_rdn0 && isfile(phi_maps_file)
        sims_need_qe_rdn0 = Int[]
        jldopen(phi_maps_file, "r") do f
            for s in sort(collect(phi_sims_done))
                haskey(f, "sim_$s/N0_rdn0") || push!(sims_need_qe_rdn0, s)
            end
        end
        if !isempty(sims_need_qe_rdn0)
            println("\nRetroactive QE RDN0: $(length(sims_need_qe_rdn0)) sims...")
            qe_Cℓn_retro = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=qe_Lmax)
            jldopen(phi_maps_file, "a+") do f
                for (i, s) in enumerate(sims_need_qe_rdn0)
                    print("\r  QE RDN0 $i/$(length(sims_need_qe_rdn0))"); flush(stdout)
                    seed_s  = haskey(f, "sim_$s/seed") ? read(f, "sim_$s/seed") : seed0 + s
                    ds_qe_s = load_sim(; seed=seed_s, load_kwargs...,
                                         bandpass_mask=qe_bandpass, Cℓn=qe_Cℓn_retro).ds
                    cl_rdn0_s = cov_to_Cℓ(N0_bias(ds_qe_s; weights=:unlensed, realization_spec=:data).N0; Δℓ=Δℓ)
                    n0_rdn0_s = Float64.(cl_rdn0_s.Cℓ)
                    f["sim_$s/N0_rdn0"]     = n0_rdn0_s
                    f["sim_$s/N0_rdn0_ell"] = Float64.(collect(cl_rdn0_s.ℓ))
                end
            end
            println("\n  Done.")
        end
    end

    # Retroactive GI RDN0 v2 (corrected formula with σ_fid)
    if isfile(phi_maps_file)
        sims_need_rdn0_v2 = Int[]
        jldopen(phi_maps_file, "r") do f
            for s in sort(collect(phi_sims_done))
                haskey(f,"sim_$s/N0_gi_rdn0_v2") || push!(sims_need_rdn0_v2, s)
            end
        end
        if !isempty(sims_need_rdn0_v2)
            println("\nGI RDN0 v2: using σ_fid_gi already computed above.")
            println("  σ_xx=$(round(σ_fid_gi.σ_xx, sigdigits=4))  σ_yy=$(round(σ_fid_gi.σ_yy, sigdigits=4))")
            println("  Retroactive GI RDN0 v2: $(length(sims_need_rdn0_v2)) sims...")
            jldopen(phi_maps_file, "a+") do f
                for (i, s) in enumerate(sims_need_rdn0_v2)
                    print("\r  RDN0-v2 $i/$(length(sims_need_rdn0_v2))"); flush(stdout)
                    seed_s = haskey(f,"sim_$s/seed") ? read(f,"sim_$s/seed") : seed0+s
                    sim_idx = seed_s - seed0
                    ds_s = load_sim(; seed=seed_s, load_kwargs...).ds
                    n0_v2 = gi_n0_rdn0(ds_s; nsim=20, σ_fid=σ_fid_gi,
                                        Lhp=4000, Lmax=Lmax, Δℓ=Δℓ,
                                        seed_start=70_000 + sim_idx * 1000)
                    f["sim_$s/N0_gi_rdn0_v2_ell"] = Float64.(n0_v2.ℓ)
                    f["sim_$s/N0_gi_rdn0_v2"]     = Float64.(n0_v2.Cℓ)
                end
            end
            println("\n  Done.")
        end
    end

    if isfile(phi_maps_file)
        sims_need_linrd = Int[]
        jldopen(phi_maps_file, "r") do f
            for s in sort(collect(phi_sims_done))
                haskey(f, "sim_$s/N0_gi_linrd") || push!(sims_need_linrd, s)
            end
        end
        if !isempty(sims_need_linrd)
            println("\nRetroactive GI lin-RD N0: $(length(sims_need_linrd)) sims...")
            jldopen(phi_maps_file, "a+") do f
                for (i, s) in enumerate(sims_need_linrd)
                    print("\r  linRD N0 $i/$(length(sims_need_linrd))"); flush(stdout)
                    seed_s  = haskey(f, "sim_$s/seed") ? read(f, "sim_$s/seed") : seed0 + s
                    sim_idx = haskey(f, "sim_$s/seed") ? (seed_s - seed0) : s
                    ds_s    = load_sim(; seed=seed_s, load_kwargs...).ds
                    n0_lr   = gi_n0_linrd_analytical(ds_s;
                        nmc=20, Lgrad=2000, Lhp=4000, Lmax=Lmax, Δℓ=Δℓ,
                        seed_start=95_000 + sim_idx * 1000,
                        smooth_C_data=true, smooth_window=9, clamp_negative=false)
                    f["sim_$s/N0_gi_linrd_ell"] = Float64.(n0_lr.ℓ)
                    f["sim_$s/N0_gi_linrd"]     = Float64.(n0_lr.Cℓ)
                end
            end
            println("\n  Done.")
        end
    end

    println("\n=== W_L diagnostics (μKarcminT=$μKarcminT, Lmax=$Lmax, beam=$(beamFWHM)′) ===")
    println("  ℓ        W_QE_RAW   W_GI_B")
    for ℓ_check in [1000, 3000, 5000, 7000, 9000, 11000]
        idx = argmin(abs.(ℓ_template .- ℓ_check))
        @printf "  ℓ≈%5d   %8.4f   %8.4f\n" round(Int, ℓ_template[idx]) W_qe_raw[idx] W_gi_b[idx]
    end
    for (name, W) in [("W_qe_raw", W_qe_raw), ("W_gi_b", W_gi_b)]
        W === nothing && continue
        msk = @. isfinite(W) & (ℓ_template >= 5000) & (ℓ_template <= 11000)
        !any(msk) && continue
        wv = W[msk]; wmin, wmax = minimum(wv), maximum(wv)
        @printf "  SANITY %-10s L=5000–11000: min=%.3f  max=%.3f  %s\n" name wmin wmax (
            wmin < -0.1 || wmax > 1.5 ? "⚠ SUSPICIOUS" : "OK")
    end
    println("Done! $nsims_completed sims.  Output: $WL_file, $phi_maps_file")

    run_map || (println("MAP section skipped."); return)

    # MAP joint
    MAP_WL_file  = "results/WL_map_12000$(map_file_suffix).jld2"
    MAP_phi_file = "results/phi_maps_map_12000$(map_file_suffix).jld2"

    map_load_kwargs = map_Lmax == Lmax ? load_kwargs : begin
        println("MAP using bandpass Lmax=$map_Lmax")
        (load_kwargs..., Cℓn=noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=map_Lmax),
                         bandpass_mask=LowPass(map_Lmax))
    end

    R_mj_sims = Vector{Vector{Float64}}(); logpdf_histories = Vector{Vector{Float64}}()
    W_mj = nothing; nsims_map_done = 0; map_seeds_done = Int[]; ℓ_template_map = nothing
    if isfile(MAP_WL_file)
        d = JLD2.load(MAP_WL_file)
        haskey(d,"R_mj_sims")       && (R_mj_sims       = d["R_mj_sims"])
        haskey(d,"logpdf_histories") && (logpdf_histories = d["logpdf_histories"])
        haskey(d,"nsims_map_done")   && (nsims_map_done   = d["nsims_map_done"])
        haskey(d,"map_seeds_done")   && (map_seeds_done   = d["map_seeds_done"])
        haskey(d,"W_mj")             && (W_mj             = d["W_mj"])
        haskey(d,"ℓ_template")       && (ℓ_template_map   = Float64.(d["ℓ_template"]))
        println("Resumed MAP checkpoint: $nsims_map_done / $map_nsims sims")
    end

    map_phi_done = Set{Int}()
    if isfile(MAP_phi_file)
        try
            jldopen(MAP_phi_file, "r") do f
                for key in keys(f)
                    m = match(r"^sim_(\d+)$", key)
                    m !== nothing && push!(map_phi_done, parse(Int, m.captures[1]))
                end
            end
            println("$(length(map_phi_done)) MAP phi maps in $MAP_phi_file")
        catch err
            @warn "MAP phi file corrupted ($err) — deleting"; rm(MAP_phi_file)
        end
    end

    map_exclude = map_extra_exclude

    for s in (nsims_map_done + 1):map_nsims
        s ∈ map_exclude && (println("  MAP Sim $s skipped"); continue)
        seed = seed0 + s
        print("  MAP Sim $s/$map_nsims (seed=$seed) ... "); flush(stdout)

        (; ϕ, ds) = load_sim(; seed=seed, map_load_kwargs...)
        if map_prior_weakening != 1.0 || map_hl_prior_weakening != 1.0
            Cϕ_conc = ds.Cϕ(;)
            new_diag = map_prior_weakening * Cϕ_conc.diag
            if map_hl_prior_weakening != 1.0
                # Build a 2D L grid matching the Fourier field layout and scale
                # C_ϕ by map_hl_prior_weakening for all modes with L > map_hl_prior_Lmin.
                ϕ_f    = FlatFourier(Cϕ_conc.diag.arr, Cϕ_conc.diag.proj)
                proj   = ϕ_f.proj
                NyF, NxF = size(ϕ_f.arr)
                ℓx2D   = repeat(proj.ℓx[1:NxF]', NyF, 1)
                ℓy2D   = repeat(proj.ℓy[1:NyF],  1,   NxF)
                L2     = @. ℓx2D^2 + ℓy2D^2
                scale  = @. ifelse(L2 > map_hl_prior_Lmin^2, Float64(map_hl_prior_weakening), 1.0)
                new_diag = FlatFourier(new_diag.arr .* scale, proj)
            end
            ds = @set ds.Cϕ = Diagonal(new_diag)
        end

        ϕ_start = if map_zero_start
            zero(ϕ)
        else
            ϕqe_ws = quadratic_estimate(ds; weights=map_warmstart_weights, wiener_filtered=true).ϕqe
            ϕ_ws   = map_warmstart_Lmax > 0 ? LowPass(map_warmstart_Lmax) * ϕqe_ws : ϕqe_ws
            map_warmstart_scale * ϕ_ws
        end

        ϕ_mj = nothing
        try
            result = MAP_joint(ds, FieldTuple(ϕ=ϕ_start);
                nsteps=map_nsteps, αmax=map_αmax,
                nburnin_update_hessian=map_nburnin_hessian,
                prior_deprojection_factor=map_prior_deproj,
                conjgrad_kwargs=(tol=map_cg_tolerance, nsteps=map_cg_nsteps), progress=false,
                history_keys=(:total_logpdf,))
            ϕ_mj = result.ϕ
            lp0, lpN = result.history[1].total_logpdf, result.history[end].total_logpdf
            @printf "(MJ lp: %.1f→%.1f) " lp0 lpN
            push!(logpdf_histories, Float64.([h.total_logpdf for h in result.history]))
        catch err
            print("MAP_joint failed($err) ")
        end

        cl_tt = get_Cℓ(ϕ; Δℓ=Δℓ)
        ℓ_template_map === nothing && (ℓ_template_map = Float64.(collect(cl_tt.ℓ)))
        ϕ_mj !== nothing && push!(R_mj_sims,
            safe_div(Float64.(get_Cℓ(ϕ, ϕ_mj; Δℓ=Δℓ).Cℓ), Float64.(cl_tt.Cℓ)))

        if s ∉ map_phi_done
            jldopen(MAP_phi_file, "a+") do f
                f["sim_$s/ϕ_true"] = Float64.(Map(ϕ).arr)
                if ϕ_mj !== nothing
                    f["sim_$s/ϕ_mj"] = Float64.(Map(ϕ_mj).arr)
                    lp_final = logpdf_histories[end][end]
                    f["sim_$s/logpdf_final"] = lp_final
                end
                f["sim_$s/seed"] = seed
            end
            push!(map_phi_done, s)
        end

        nsims_map_done += 1; push!(map_seeds_done, seed)
        # Use per-bin median: robust against rare realizations where C_true(ℓ)≈0 in a
        # narrow bin sends the cross/true ratio to infinity and corrupts the arithmetic mean.
        W_mj = !isempty(R_mj_sims) ? [median(R[i] for R in R_mj_sims) for i in 1:length(R_mj_sims[1])] : nothing
        jldsave(MAP_WL_file; R_mj_sims, logpdf_histories, W_mj, nsims_map_done,
                map_seeds_done, ℓ_template=ℓ_template_map)
        println("done"); flush(stdout)
    end

    if !isempty(R_mj_sims)
        println("\n=== MAP W_L diagnostics (μKarcminT=$μKarcminT) ===")
        for ℓ_check in [3000, 5000, 7000, 9000, 11000]
            idx = argmin(abs.(ℓ_template_map .- ℓ_check))
            @printf "  ℓ≈%5d   %8.4f\n" round(Int, ℓ_template_map[idx]) (W_mj !== nothing ? W_mj[idx] : NaN)
        end
        println("MAP done! $nsims_map_done/$nsims_map sims.  Output: $MAP_WL_file, $MAP_phi_file")
    end

    # Convergence summary from in-memory logpdf_histories.
    # -Inf outliers (failed MAP_joint calls) are filtered before averaging.
    good_hists = filter(h -> all(isfinite, h), logpdf_histories)
    if length(good_hists) >= 2
        nsteps_done = minimum(length.(good_hists))
        mean_lp  = [mean(h[t] for h in good_hists) for t in 1:nsteps_done]
        total_Δ  = mean_lp[end] - mean_lp[1]
        last5_Δ  = nsteps_done >= 5 ? mean_lp[end] - mean_lp[end-4] : NaN
        per_step = total_Δ / (nsteps_done - 1)
        n_bad    = length(logpdf_histories) - length(good_hists)
        println("\n  MAP convergence — logpdf ($(length(good_hists)) sims$(n_bad > 0 ? ", $n_bad outlier(s) excluded" : "")):")
        @printf "    Step 1:  %.2f\n" mean_lp[1]
        @printf "    Step %d: %.2f  (Δ=%.2f,  %.3f/step)\n" nsteps_done mean_lp[end] total_Δ per_step
        isfinite(last5_Δ) && @printf "    Last 5 steps Δ = %.4f\n" last5_Δ
        if isfinite(last5_Δ) && abs(per_step) > 0
            frac = abs(last5_Δ/5) / abs(per_step)
            println(frac > 0.05 ?
                "    ⚠ STILL CONVERGING (last-step rate=$(round(100*frac;digits=1))% of mean)" :
                "    ✓ CONVERGED (last-step rate < 5% of mean)")
        end

        if run_convergence_diag
            diag_file = "results/diag_map_convergence$(map_file_suffix).jld2"
            jldsave(diag_file; mean_logpdf=mean_lp, all_logpdf_histories=logpdf_histories,
                    good_logpdf_histories=good_hists, map_file_suffix, μKarcminT, map_nsteps)
            println("  Saved convergence diag → $diag_file")
        end
    end

end


# run_gi_wiener: pixel-exact Wiener GI (gi_estimate_corrected).
# Uncomment the call below to generate results; output is not loaded by plot_results.jl.
function run_gi_wiener(μKarcminT::Float64, suffix::String;
                       nsims_w::Int=100, Lmax::Int=12000, beamFWHM::Float64=0.3,
                       Δℓ_wl::Int=150)
    WL_file  = "results/WL_gi_wiener_12000$(suffix).jld2"
    phi_file = "results/phi_maps_gi_wiener_12000$(suffix).jld2"
    println("\n" * "="^70)
    println("=== GI-Wiener  μKarcminT=$μKarcminT  Lmax=$Lmax  suffix=\"$suffix\" ===")
    println("    WL:  $WL_file\n    phi: $phi_file")
    println("="^70)

    Cℓn = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
    lkw = (Cℓ=Cℓ, Cℓn=Cℓn, θpix=θpix, T=Float64, Nside=Nside,
           beamFWHM=beamFWHM, pol=pol, bandpass_mask=LowPass(Lmax),
           pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0))

    safe_div(a, b) = @. ifelse(abs(b) > 0.0, a / b, 0.0)

    # resume
    sum_R = W_giw = ℓ_tmpl = nothing
    nsims_done = 0; seeds_done = Int[]
    if isfile(WL_file)
        d = JLD2.load(WL_file)
        lf(k) = (haskey(d,k) && d[k] !== nothing) ? Float64.(d[k]) : nothing
        sum_R = lf("sum_R"); W_giw = lf("W_giw"); ℓ_tmpl = lf("ℓ_template")
        haskey(d,"nsims_done")  && (nsims_done  = d["nsims_done"])
        haskey(d,"seeds_done")  && (seeds_done  = d["seeds_done"])
        println("Resumed: $nsims_done / $nsims_w sims")
    end
    phi_done = Set{Int}()
    if isfile(phi_file)
        try
            jldopen(phi_file, "r") do f
                for k in keys(f)
                    m = match(r"^sim_(\d+)$", k); m !== nothing && push!(phi_done, parse(Int, m.captures[1]))
                end
            end
            println("$(length(phi_done)) phi maps already in $phi_file")
        catch err
            @warn "phi_file corrupted ($err) — deleting"; rm(phi_file)
        end
    end

    for s in (nsims_done + 1):nsims_w
        seed = seed0 + s
        print("  GIW sim $s/$nsims_w (seed=$seed) ... "); flush(stdout)
        (; ϕ, ds) = load_sim(; seed=seed, lkw...)

        ϕgiw = gi_estimate_corrected(ds; Lhp=4000, Lmax=Lmax)

        cl_tt  = get_Cℓ(ϕ;      Δℓ=Δℓ_wl)
        cl_tgw = get_Cℓ(ϕ, ϕgiw; Δℓ=Δℓ_wl)

        if ℓ_tmpl === nothing
            ℓ_tmpl = Float64.(collect(cl_tt.ℓ))
            sum_R  = zeros(Float64, length(ℓ_tmpl))
        end
        sum_R .+= safe_div(Float64.(cl_tgw.Cℓ), Float64.(cl_tt.Cℓ))
        nsims_done += 1; push!(seeds_done, seed)
        W_giw = sum_R ./ nsims_done

        jldopen(phi_file, "a+") do f
            if s ∉ phi_done
                f["sim_$s/ϕ_true"]  = Float64.(Map(ϕ).arr)
                f["sim_$s/ϕ_gi_w"]  = Float64.(Map(ϕgiw).arr)
                f["sim_$s/seed"]    = seed
                push!(phi_done, s)
            end
        end
        @save WL_file sum_R W_giw ℓ_template=ℓ_tmpl nsims_done seeds_done
        println("done"); flush(stdout)
    end
    println("GI-Wiener done! $nsims_done sims.  Output: $WL_file, $phi_file")
end

# run_gi_wiener(0.1, "_ul")   # uncomment to run pixel-exact GI (slow)

# S4-like: 1 µK-arcmin, 1' beam
run_noise_level(1.0, "", 12000, 1.0;
                run_rdn0=true, run_map=true,
                map_nsteps=40, map_αmax=0.3,
                map_cg_nsteps=200, map_nsims=500)

# Ultra-low noise: 0.1 µK-arcmin, 0.3' beam
run_noise_level(0.1, "_ul", 12000, 0.3;
                Δℓ_wl=150,
                run_rdn0=false,
                run_map=true,
                map_file_suffix="_ul_zero_a01",
                map_zero_start=true,
                map_αmax=0.05,
                map_nburnin_hessian=typemax(Int),
                map_nsteps=70,
                map_nsims=500, map_extra_exclude=Set{Int}([100,102]))

println("\nAll done.")

