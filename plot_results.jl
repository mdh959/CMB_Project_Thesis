#!/usr/bin/env julia
# plot_results.jl
# Figures: mean spectra, σ panels (auto/cross), W_L, ρ_L, covariance matrices,
#          MAP convergence, SNR table.
import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
import PythonCall
using PythonCall: pyimport, pyconvert
using Statistics: mean, std, median, quantile
using PythonPlot
using Printf
using JLD2
using HDF5

include("utils.jl")
using .Utils

const Cℓ_theory   = camb(r=0.05, ℓmax=35000)

const θpix        = 0.7438046267475303
const Nside       = 512
const pol         = :I
const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (Nside * θpix_rad)^2 / (4π)
const f_sky_paper = 0.4           # Hadzhiyska+2019 survey sky fraction

const minW        = 1e-8
const minW_auto   = 0.5
const minW_auto_gi = 0.2

const ΔL          = 2000          # bandpower bin width (matching Hadzhiyska+2019)
const Δℓ_spec     = 30
const proc_edges  = collect(3000.0:ΔL:13001.0)
const OUT_DIR     = "results"

mkpath(OUT_DIR)

function smooth_wl(W::Vector{Float64}; window::Int=9)
    n = length(W); Ws = similar(W); hw = window ÷ 2
    for i in 1:n
        vals = filter(isfinite, W[max(1,i-hw):min(n,i+hw)])
        Ws[i] = isempty(vals) ? 0.0 : mean(vals)
    end
    return Ws
end

function coarsen(ℓ::Vector, M::Matrix; edges::Vector)
    nb = length(edges) - 1
    Lc = fill(NaN, nb); μv = fill(NaN, nb); σv = fill(NaN, nb)
    for b in 1:nb
        idx = findall(x -> edges[b] <= x < edges[b+1], ℓ)
        isempty(idx) && continue
        per = filter(isfinite, vec(mean(M[idx, :]; dims=1)))
        length(per) < 2 && continue
        Lc[b] = 0.5*(edges[b]+edges[b+1]); μv[b] = mean(per); σv[b] = std(per)
    end
    Lc, μv, σv
end

function bands_per_sim(ℓ::Vector, sims::Vector, edges::Vector)
    nb = length(edges) - 1; ns = length(sims)
    ns == 0 && return fill(NaN, nb, 0)
    M = fill(NaN, nb, ns)
    for (j, s) in enumerate(sims), b in 1:nb
        idx = findall(x -> edges[b] <= x < edges[b+1], ℓ)
        isempty(idx) && continue
        vals = filter(isfinite, s[idx])
        isempty(vals) || (M[b, j] = mean(vals))
    end
    M
end

function snr_compute(M_meas::Matrix, M_true::Matrix, Lc::Vector,
                     L_lo::Real, L_hi::Real, σ_sc::Real)
    size(M_meas, 2) < 2 && return NaN
    sel = findall(b -> !isnan(Lc[b]) && L_lo <= Lc[b] <= L_hi, 1:length(Lc))
    isempty(sel) && return NaN
    snr2 = 0.0; nb = 0
    for b in sel
        a_b = filter(isfinite, vec(M_meas[b,:])); t_b = filter(isfinite, vec(M_true[b,:]))
        (length(a_b) < 2 || length(t_b) < 2) && continue
        mean_t = mean(t_b)
        abs(mean_t) > 1e-30 && abs(mean(a_b)) < 0.05 * abs(mean_t) && continue
        σ_b = σ_sc * std(a_b); σ_b <= 0 && continue
        snr2 += (mean_t / σ_b)^2; nb += 1
    end
    nb == 0 && return NaN
    sqrt(max(0.0, snr2))
end

function process_noise_level(WL_file, phi_maps_file, label;
                             Lmax=12000, beamFWHM=1.0, μKarcminT=1.0,
                             map_wl_file=nothing, map_phi_file=nothing,
                             snr_L_lo=5000.0, snr_L_hi=11000.0,
                             xlim_plot=(4000.0, 12000.0),
                             θpix_sim::Float64=θpix, Nside_sim::Int=Nside,
                             exclude_sims::Set{Int}=Set{Int}(),
                             map_exclude_sims::Set{Int}=Set{Int}(),
                             gi_rms_outliers::Bool=false,
                             qe_snr_use_all::Bool=false,
                             meta_phi_source::Union{String,Nothing}=nothing,
                             mj_rms_max::Float64=1e-4,
                             mj_bp_outliers::Bool=false,
                             use_hdf5::Bool=false,
                             linrd_also_subtract_fgmc::Bool=false)
    println("\n" * "="^70)
    println("=== $label ===")

    d_wl       = JLD2.load(WL_file)
    ℓ_template = Float64.(d_wl["ℓ_template"])
    nsims_wl   = d_wl["nsims_completed"]
    _lw(d,k)   = (haskey(d,k) && d[k] !== nothing) ? Float64.(d[k]) : nothing

    W_qe_raw = _lw(d_wl, "W_qe_raw")
    W_qe_raw === nothing && (W_qe_raw = _lw(d_wl, "W_qe_wf"))
    W_gi_b   = _lw(d_wl, "W_gi_b")

    W_qe_s = W_qe_raw !== nothing ? smooth_wl(W_qe_raw) : nothing
    W_gi_s = W_gi_b   !== nothing ? smooth_wl(W_gi_b)   : nothing
    WL_qe  = W_qe_s !== nothing ? Cℓs(ℓ_template, W_qe_s) : nothing
    WL_gi  = W_gi_s !== nothing ? Cℓs(ℓ_template, W_gi_s) : nothing
    has_qe = WL_qe !== nothing
    has_gi = WL_gi !== nothing
    println("  W_L from $nsims_wl sims  (QE: $has_qe, GI: $has_gi)")

    has_map = map_wl_file !== nothing && isfile(map_wl_file) &&
              map_phi_file !== nothing && isfile(map_phi_file)
    W_mj_raw = WL_mj = nothing
    ℓ_template_map = ℓ_template
    nsims_map_done = 0
    if has_map
        d_map = JLD2.load(map_wl_file)
        ℓ_template_map = haskey(d_map, "ℓ_template") ?
            Float64.(d_map["ℓ_template"]) : ℓ_template
        W_mj_raw = haskey(d_map, "W_mj") ?
            Float64.(d_map["W_mj"]) : ones(length(ℓ_template_map))
        nsims_map_done = get(d_map, "nsims_map_done", 0)
        WL_mj  = Cℓs(ℓ_template_map, smooth_wl(W_mj_raw))
        println("  MAP W_L: $nsims_map_done sims")
    end

    Cℓn_meta  = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
    _meta_file = meta_phi_source !== nothing ? meta_phi_source : phi_maps_file
    meta_seed = jldopen(_meta_file, "r") do f
        first_sim = minimum(parse(Int, match(r"^sim_(\d+)$", k).captures[1])
                            for k in keys(f) if occursin(r"^sim_\d+$", k))
        read(f, "sim_$first_sim/seed")
    end
    (; ϕ) = load_sim(; seed=meta_seed, Cℓ=Cℓ_theory, Cℓn=Cℓn_meta, θpix=θpix_sim,
                       T=Float64, Nside=Nside_sim, beamFWHM=beamFWHM, pol=pol,
                       bandpass_mask=LowPass(Lmax),
                       pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0))
    ϕ_ref = Map(ϕ)
    wrap(arr) = typeof(ϕ_ref)(Float64.(arr), ϕ_ref.metadata)

    Cl_auto_qe_sims      = Vector{Vector{Float64}}()
    Cl_auto_qe_full_sims = Vector{Vector{Float64}}()
    Cl_auto_qe_rdn0_sims = Vector{Vector{Float64}}()
    Cl_cross_qe_sims     = Vector{Vector{Float64}}()
    Cl_auto_gi_sims      = Vector{Vector{Float64}}()
    Cl_auto_gi_full_sims = Vector{Vector{Float64}}()
    Cl_auto_gi_fgmc_sims = Vector{Vector{Float64}}()
    Cl_auto_gi_rdn0_sims = Vector{Vector{Float64}}()
    Cl_auto_gi_linrd_sims = Vector{Vector{Float64}}()
    Cl_cross_gi_sims     = Vector{Vector{Float64}}()
    Cl_true_sims         = Vector{Vector{Float64}}()
    ρ_qe_sims            = Vector{Vector{Float64}}()
    ρ_gi_sims            = Vector{Vector{Float64}}()
    ρ_mj_gi_sims         = Vector{Vector{Float64}}()
    ρ_mj_qe_sims         = Vector{Vector{Float64}}()
    ρ_gi_qe_sims         = Vector{Vector{Float64}}()
    W_qe_per_sim_sims    = Vector{Vector{Float64}}()
    W_gi_per_sim_sims    = Vector{Vector{Float64}}()
    ℓ_kk                 = nothing

    qegi_sims  = Int[]
    mean_N0_qe = nothing

    _phi_open(body, fname) = use_hdf5 ? HDF5.h5open(body, fname, "r") : jldopen(body, fname, "r")

    _phi_open(phi_maps_file) do f
        for key in keys(f)
            m = match(r"^sim_(\d+)$", key)
            m !== nothing && push!(qegi_sims, parse(Int, m.captures[1]))
        end
        sort!(qegi_sims)

        _flat_f = !isempty(qegi_sims) && (try f["sim_$(qegi_sims[1])"] isa AbstractDict catch; false end)
        _sim_cache_f = Dict{Int,Any}()
        _load_f(s) = get!(_sim_cache_f, s) do; haskey(f, "sim_$s") ? f["sim_$s"] : Dict{String,Any}(); end
        _hk(s, k) = _flat_f ? haskey(_load_f(s), k) : haskey(f, "sim_$s/$k")
        _rd(s, k) = _flat_f ? _load_f(s)[k] : read(f, "sim_$s/$k")

        if has_map
            map_sims_available = Set{Int}()
            _phi_open(map_phi_file) do fm
                for key in keys(fm)
                    m2 = match(r"^sim_(\d+)$", key)
                    m2 !== nothing && push!(map_sims_available, parse(Int, m2.captures[1]))
                end
            end
            filter!(s -> s ∈ map_sims_available, qegi_sims)
            println("  QE/GI sims restricted to $(length(qegi_sims)) matching MAP")
        end

        !isempty(exclude_sims) && (filter!(s -> s ∉ exclude_sims, qegi_sims);
            println("  Excluded sims: $(sort(collect(exclude_sims)))"))

        if gi_rms_outliers && has_gi
            gi_rms_vals = Dict{Int,Float64}()
            for s in qegi_sims
                _hk(s, "ϕ_gi_b") || continue
                gi_rms_vals[s] = sqrt(mean(Map(wrap(_rd(s, "ϕ_gi_b"))).arr .^ 2))
            end
            if !isempty(gi_rms_vals)
                med_gi = median(values(gi_rms_vals))
                bad_gi = Set(s for (s, r) in gi_rms_vals if r > 15.0 * med_gi)
                !isempty(bad_gi) && (println("  GI RMS outliers: $(sort(collect(bad_gi)))");
                    filter!(s -> s ∉ bad_gi, qegi_sims))
            end
        end

        for (i, s) in enumerate(qegi_sims)
            ϕt = wrap(_rd(s, "ϕ_true"))
            cl_tt = get_Cℓ(ϕt; Δℓ=Δℓ_spec)
            ℓ_kk === nothing && (ℓ_kk = Float64.(collect(cl_tt.ℓ)))
            kfac = @. (ℓ_kk^2 / 2)^2
            push!(Cl_true_sims, kfac .* Float64.(cl_tt.Cℓ))

            _ϕr_deb_iter = nothing; _cl_aa_iter = nothing

            if has_qe && _hk(s, "ϕ_qe_raw")
                ϕr     = wrap(_rd(s, "ϕ_qe_raw"))
                push!(W_qe_per_sim_sims, smooth_wl(Float64.(get_Cℓ(ϕt, ϕr; Δℓ=Δℓ_spec).Cℓ) ./ max.(Float64.(cl_tt.Cℓ), 1e-60)))
                ϕr_deb = debias_phi_with_WL(ϕr, WL_qe; minW=minW)
                cl_aa  = get_Cℓ(ϕr_deb; Δℓ=Δℓ_spec)
                cl_xa  = get_Cℓ(ϕt, ϕr_deb; Δℓ=Δℓ_spec)
                Cl_aa_vals = Float64.(cl_aa.Cℓ)
                W_at_ℓkk   = Float64.(WL_qe.(ℓ_kk))
                W2_floor    = @. max(W_at_ℓkk^2, minW_auto^2)
                push!(Cl_auto_qe_full_sims, kfac .* Cl_aa_vals)
                if _hk(s, "N0_rdn0")
                    N0_ps_raw = Float64.(_rd(s, "N0_rdn0"))
                    N0_ps_raw = [isfinite(v) && v >= 0.0 ? v : 0.0 for v in N0_ps_raw]
                    if !isempty(N0_ps_raw)
                        n0_ps_ell = _hk(s, "N0_rdn0_ell") ?
                                    Float64.(_rd(s, "N0_rdn0_ell")) : ℓ_kk
                        if length(N0_ps_raw) == length(n0_ps_ell)
                            itp_ps = Cℓs(n0_ps_ell, N0_ps_raw)
                            N0_ps = [ℓ < n0_ps_ell[1] || ℓ > n0_ps_ell[end] ? 0.0 :
                                     max(Float64(itp_ps(ℓ)), 0.0) for ℓ in ℓ_kk]
                            push!(Cl_auto_qe_rdn0_sims,
                                  kfac .* (@. Cl_aa_vals - N0_ps / W2_floor))
                        end
                    end
                end
                N0_mean_deb = (mean_N0_qe !== nothing && length(mean_N0_qe) == length(ℓ_kk)) ?
                              mean_N0_qe ./ W2_floor : zeros(length(ℓ_kk))
                push!(Cl_auto_qe_sims, kfac .* (Cl_aa_vals .- N0_mean_deb))
                push!(Cl_cross_qe_sims, kfac .* Float64.(cl_xa.Cℓ))
                denom = sqrt.(max.(Float64.(cl_tt.Cℓ) .* max.(Cl_aa_vals, 0.0), 0.0))
                push!(ρ_qe_sims, clamp.(Float64.(cl_xa.Cℓ) ./ max.(denom, 1e-30), -1.0, 1.0))
                _ϕr_deb_iter = ϕr_deb; _cl_aa_iter = cl_aa
            end

            if has_gi && _hk(s, "ϕ_gi_b")
                ϕg_raw = wrap(_rd(s, "ϕ_gi_b"))
                push!(W_gi_per_sim_sims, smooth_wl(Float64.(get_Cℓ(ϕt, ϕg_raw; Δℓ=Δℓ_spec).Cℓ) ./ max.(Float64.(cl_tt.Cℓ), 1e-60)))
                ϕg_F   = Fourier(ϕg_raw)
                ϕg_deb = debias_phi_with_WL(ϕg_F, WL_gi; minW=minW)
                cl_gi_a      = get_Cℓ(ϕg_deb; Δℓ=Δℓ_spec)
                cl_gi_a_vals = Float64.(cl_gi_a.Cℓ)
                cl_gi_r      = get_Cℓ(ϕg_F; Δℓ=Δℓ_spec)
                cl_gi_r_vals = Float64.(cl_gi_r.Cℓ)
                W_gi_at_ℓkk  = Float64.(WL_gi.(ℓ_kk))
                W2_gi_floor  = @. max(W_gi_at_ℓkk^2, minW_auto_gi^2)
                push!(Cl_auto_gi_full_sims, kfac .* cl_gi_a_vals)
                push!(Cl_auto_gi_sims,      kfac .* cl_gi_a_vals)
                _N0_fg_gi = zeros(length(ℓ_kk))
                if _hk(s, "N0_gi_fgmc")
                    N0_fgmc_ell  = Float64.(_rd(s, "N0_gi_fgmc_ell"))
                    N0_fgmc_raw  = Float64.(_rd(s, "N0_gi_fgmc"))
                    N0_fgmc_safe = [isfinite(v) && v >= 0.0 ? v : 0.0 for v in N0_fgmc_raw]
                    itp_fg       = Cℓs(N0_fgmc_ell, N0_fgmc_safe)
                    N0_fg_at_kk  = [ℓ < N0_fgmc_ell[1] || ℓ > N0_fgmc_ell[end] ? 0.0 :
                                    max(Float64(itp_fg(ℓ)), 0.0) for ℓ in ℓ_kk]
                    _N0_fg_gi    = N0_fg_at_kk
                    push!(Cl_auto_gi_fgmc_sims,
                          kfac .* (@. (cl_gi_r_vals - N0_fg_at_kk) / W2_gi_floor))
                end
                if _hk(s, "N0_gi_rdn0_v2")
                    N0_rd_ell   = Float64.(_rd(s, "N0_gi_rdn0_v2_ell"))
                    N0_rd_raw   = Float64.(_rd(s, "N0_gi_rdn0_v2"))
                    itp_rd      = Cℓs(N0_rd_ell, N0_rd_raw)
                    N0_rd_at_kk = [ℓ < N0_rd_ell[1] || ℓ > N0_rd_ell[end] ? 0.0 :
                                   Float64(itp_rd(ℓ)) for ℓ in ℓ_kk]
                    push!(Cl_auto_gi_rdn0_sims,
                          kfac .* (@. (cl_gi_r_vals - N0_rd_at_kk) / W2_gi_floor))
                end
                if _hk(s, "N0_gi_linrd")
                    N0_lr_ell   = Float64.(_rd(s, "N0_gi_linrd_ell"))
                    N0_lr_raw   = Float64.(_rd(s, "N0_gi_linrd"))
                    itp_lr      = Cℓs(N0_lr_ell, N0_lr_raw)
                    N0_lr_at_kk = [ℓ < N0_lr_ell[1] || ℓ > N0_lr_ell[end] ? 0.0 :
                                   Float64(itp_lr(ℓ)) for ℓ in ℓ_kk]
                    N0_lr_total = linrd_also_subtract_fgmc ? N0_lr_at_kk .+ _N0_fg_gi : N0_lr_at_kk
                    push!(Cl_auto_gi_linrd_sims,
                          kfac .* (@. (cl_gi_r_vals - N0_lr_total) / W2_gi_floor))
                end
                cl_gi_x = get_Cℓ(ϕt, ϕg_deb; Δℓ=Δℓ_spec)
                push!(Cl_cross_gi_sims, kfac .* Float64.(cl_gi_x.Cℓ))
                denom = sqrt.(max.(Float64.(cl_tt.Cℓ) .* max.(cl_gi_a_vals, 0.0), 0.0))
                push!(ρ_gi_sims, clamp.(Float64.(cl_gi_x.Cℓ) ./ max.(denom, 1e-30), -1.0, 1.0))
                if _ϕr_deb_iter !== nothing
                    cl_gq    = get_Cℓ(ϕg_deb, _ϕr_deb_iter; Δℓ=Δℓ_spec)
                    denom_gq = sqrt.(max.(cl_gi_a_vals .* Float64.(_cl_aa_iter.Cℓ), 0.0))
                    push!(ρ_gi_qe_sims, clamp.(Float64.(cl_gq.Cℓ) ./ max.(denom_gq, 1e-30), -1.0, 1.0))
                end
            end
            print("\r  QE/GI sim $i/$(length(qegi_sims))"); flush(stdout)
        end
    end
    println()
    println("  $(length(qegi_sims)) QE/GI sims")
    println("    QE RDN0  : $(length(Cl_auto_qe_rdn0_sims))")
    println("    GI fg-MC : $(length(Cl_auto_gi_fgmc_sims))")
    println("    GI RDN0  : $(length(Cl_auto_gi_rdn0_sims))")
    println("    GI lin-RD: $(length(Cl_auto_gi_linrd_sims))")

    Cl_auto_mj_sims  = Vector{Vector{Float64}}()
    Cl_cross_mj_sims = Vector{Vector{Float64}}()
    Cl_true_mj_sims  = Vector{Vector{Float64}}()
    ρ_mj_sims        = Vector{Vector{Float64}}()
    W_mj_per_sim_sims = Vector{Vector{Float64}}()
    if has_map && ℓ_kk !== nothing
        map_sims = Int[]
        n_auto_excl = 0
        lp_by_sim = Dict{Int,Float64}()
        _phi_open(map_phi_file) do f
            for key in keys(f)
                m = match(r"^sim_(\d+)$", key); m !== nothing && push!(map_sims, parse(Int, m.captures[1]))
            end
            sort!(map_sims)
            _flat_m = !isempty(map_sims) && (try f["sim_$(map_sims[1])"] isa AbstractDict catch; false end)
            _sim_cache_m = Dict{Int,Any}()
            _load_m(s) = get!(_sim_cache_m, s) do; haskey(f, "sim_$s") ? f["sim_$s"] : Dict{String,Any}(); end
            _hkm(s, k) = _flat_m ? haskey(_load_m(s), k) : haskey(f, "sim_$s/$k")
            _rdm(s, k) = _flat_m ? _load_m(s)[k] : read(f, "sim_$s/$k")
            d_map_tmp = JLD2.load(map_wl_file)
            if haskey(d_map_tmp, "logpdf_histories")
                phi_mj_sims = sort([s for s in map_sims if _hkm(s, "ϕ_mj")])
                hists = d_map_tmp["logpdf_histories"]
                for (i, s) in enumerate(phi_mj_sims)
                    i > length(hists) && break
                    h = hists[i]
                    lp_by_sim[s] = isempty(h) ? NaN : Float64(h[end])
                end
            end
            !isempty(map_exclude_sims) && (filter!(s -> s ∉ map_exclude_sims, map_sims);
                println("  MAP manually excluded: $(sort(collect(map_exclude_sims)))"))
            _phi_open(phi_maps_file) do fqegi
                _flat_q = !isempty(qegi_sims) && (try fqegi["sim_$(qegi_sims[1])"] isa AbstractDict catch; false end)
                _sim_cache_q = Dict{Int,Any}()
                _load_q(sx) = get!(_sim_cache_q, sx) do; haskey(fqegi, "sim_$sx") ? fqegi["sim_$sx"] : Dict{String,Any}(); end
                _hkq(sx, k) = _flat_q ? haskey(_load_q(sx), k) : haskey(fqegi, "sim_$sx/$k")
                _rdq(sx, k) = _flat_q ? _load_q(sx)[k] : read(fqegi, "sim_$sx/$k")
                for (i, s) in enumerate(map_sims)
                    _hkm(s, "ϕ_mj") || continue
                    ϕt   = wrap(_rdm(s, "ϕ_true"))
                    ϕm_r = wrap(_rdm(s, "ϕ_mj"))
                    ϕm_arr = Float64.(Map(ϕm_r).arr)
                    mj_rms = sqrt(mean(ϕm_arr .^ 2))
                    lp = _hkm(s, "logpdf_final") ?
                        Float64(_rdm(s, "logpdf_final")) : get(lp_by_sim, s, NaN)
                    lp_bad = !isnan(lp) && (!isfinite(lp) || lp < -1e10)
                    if !all(isfinite, ϕm_arr) || mj_rms > mj_rms_max || lp_bad
                        n_auto_excl += 1; continue
                    end
                    ϕm_d = debias_phi_with_WL(ϕm_r, WL_mj; minW=minW)
                    kfac = @. (ℓ_kk^2 / 2)^2
                    cl_a = get_Cℓ(ϕm_d; Δℓ=Δℓ_spec)
                    cl_x = get_Cℓ(ϕt, ϕm_d; Δℓ=Δℓ_spec)
                    push!(Cl_auto_mj_sims,  kfac .* Float64.(cl_a.Cℓ))
                    push!(Cl_cross_mj_sims, kfac .* Float64.(cl_x.Cℓ))
                    cl_tt  = get_Cℓ(ϕt; Δℓ=Δℓ_spec)
                    push!(Cl_true_mj_sims, kfac .* Float64.(cl_tt.Cℓ))
                    push!(W_mj_per_sim_sims, smooth_wl(Float64.(get_Cℓ(ϕt, ϕm_r; Δℓ=Δℓ_spec).Cℓ) ./ max.(Float64.(cl_tt.Cℓ), 1e-60)))
                    denom  = sqrt.(max.(Float64.(cl_tt.Cℓ) .* Float64.(cl_a.Cℓ), 0.0))
                    push!(ρ_mj_sims, clamp.(Float64.(cl_x.Cℓ) ./ max.(denom, 1e-30), -1.0, 1.0))
                    if has_gi && _hkq(s, "ϕ_gi_b")
                        ϕg_r  = wrap(_rdq(s, "ϕ_gi_b"))
                        ϕg_d  = debias_phi_with_WL(ϕg_r, WL_gi; minW=minW)
                        cl_mg = get_Cℓ(ϕm_d, ϕg_d; Δℓ=Δℓ_spec)
                        cl_g  = get_Cℓ(ϕg_d; Δℓ=Δℓ_spec)
                        denom_mg = sqrt.(max.(Float64.(cl_a.Cℓ) .* Float64.(cl_g.Cℓ), 0.0))
                        push!(ρ_mj_gi_sims, clamp.(Float64.(cl_mg.Cℓ) ./ max.(denom_mg, 1e-30), -1.0, 1.0))
                    end
                    if has_qe && _hkq(s, "ϕ_qe_raw")
                        ϕq_r  = wrap(_rdq(s, "ϕ_qe_raw"))
                        ϕq_d  = debias_phi_with_WL(ϕq_r, WL_qe; minW=minW)
                        cl_mq = get_Cℓ(ϕm_d, ϕq_d; Δℓ=Δℓ_spec)
                        cl_q  = get_Cℓ(ϕq_d; Δℓ=Δℓ_spec)
                        denom_mq = sqrt.(max.(Float64.(cl_a.Cℓ) .* Float64.(cl_q.Cℓ), 0.0))
                        push!(ρ_mj_qe_sims, clamp.(Float64.(cl_mq.Cℓ) ./ max.(denom_mq, 1e-30), -1.0, 1.0))
                    end
                    print("\r  MAP sim $i/$(length(map_sims))"); flush(stdout)
                end
            end
        end
        println(); println("  $(length(Cl_auto_mj_sims)) MAP sims (auto-excluded $n_auto_excl)")
    end

    θpr        = θpix_sim * π / (180 * 60)
    f_sky_sim  = (Nside_sim * θpr)^2 / (4π)
    σ_sc       = sqrt(f_sky_sim / f_sky_paper)
    ℓv = ℓ_kk !== nothing ? ℓ_kk : Float64[]

    Cl_auto_qe_rdn0_filt = let
        if !isempty(Cl_auto_qe_rdn0_sims) && !isempty(ℓv)
            B = bands_per_sim(ℓv, Cl_auto_qe_rdn0_sims, proc_edges)
            sim_mag = vec(mean(abs.(B); dims=1))
            med = median(filter(isfinite, sim_mag))
            keep = findall(j -> isfinite(sim_mag[j]) && sim_mag[j] <= 15.0 * max(med, 1e-100), 1:length(Cl_auto_qe_rdn0_sims))
            n_rem = length(Cl_auto_qe_rdn0_sims) - length(keep)
            n_rem > 0 && println("  QE RDN0 outlier filter: removed $n_rem sims")
            Cl_auto_qe_rdn0_sims[keep]
        else
            Cl_auto_qe_rdn0_sims
        end
    end

    Cl_auto_mj_filt, Cl_cross_mj_filt = Cl_auto_mj_sims, Cl_cross_mj_sims
    if mj_bp_outliers && !isempty(Cl_auto_mj_filt) && !isempty(ℓv)
        B = bands_per_sim(ℓv, Cl_auto_mj_filt, proc_edges)
        sim_mag = vec(mean(abs.(B); dims=1))
        med = median(filter(isfinite, sim_mag))
        keep = findall(j -> isfinite(sim_mag[j]) && sim_mag[j] <= 15.0 * max(med, 1e-100), 1:length(Cl_auto_mj_filt))
        n_rem = length(Cl_auto_mj_filt) - length(keep)
        n_rem > 0 && println("  MAP outlier filter: removed $n_rem sims")
        Cl_auto_mj_filt  = Cl_auto_mj_filt[keep]
        Cl_cross_mj_filt = Cl_cross_mj_filt[keep]
    end

    function _gi_bp_filt(sims, name)
        (isempty(sims) || isempty(ℓv)) && return sims
        B = bands_per_sim(ℓv, sims, proc_edges)
        sim_mag = vec(mean(abs.(B); dims=1))
        med = median(filter(isfinite, sim_mag))
        keep = findall(j -> isfinite(sim_mag[j]) && sim_mag[j] <= 15.0 * max(med, 1e-100), 1:length(sims))
        n_rem = length(sims) - length(keep)
        n_rem > 0 && println("  GI $name outlier filter: removed $n_rem sims")
        sims[keep]
    end
    Cl_auto_gi_fgmc_filt  = _gi_bp_filt(Cl_auto_gi_fgmc_sims,  "fgmc")
    Cl_auto_gi_rdn0_filt  = _gi_bp_filt(Cl_auto_gi_rdn0_sims,  "rdn0")
    Cl_auto_gi_linrd_filt = _gi_bp_filt(Cl_auto_gi_linrd_sims, "linrd")

    Tmat = reduce(hcat, Cl_true_sims)
    Lc, C̄_true, _ = coarsen(ℓv, Tmat; edges=proc_edges)
    Neff_v = @. (2*Lc + 1) * ΔL * f_sky_paper

    function proc(sa, sa_full, sx)
        nb = length(Lc)
        z = fill(NaN, nb)
        isempty(sa_full) && return z,z,z,z,z,z
        A_full = reduce(hcat, sa_full)
        _, C̄a_full, _ = coarsen(ℓv, A_full; edges=proc_edges)
        C̄a = z; σa = z
        if !isempty(sa)
            _, C̄a, σa = coarsen(ℓv, reduce(hcat, sa); edges=proc_edges); σa .*= σ_sc
        end
        C̄x = z; σx = z
        if !isempty(sx)
            _, C̄x, σx = coarsen(ℓv, reduce(hcat, sx); edges=proc_edges); σx .*= σ_sc
        end
        σ_th_a = @. sqrt(2 / Neff_v) * abs(C̄a_full)
        σ_th_x = @. sqrt(max(abs(C̄_true)*abs(C̄a_full) + C̄x^2, 0.0) / Neff_v)
        C̄a, σa, σ_th_a, C̄x, σx, σ_th_x
    end

    _qe_sa_all  = isempty(Cl_auto_qe_rdn0_sims) ? Cl_auto_qe_sims : Cl_auto_qe_rdn0_sims
    _qe_sa_filt = isempty(Cl_auto_qe_rdn0_filt) ? Cl_auto_qe_sims : Cl_auto_qe_rdn0_filt
    _, σ_a_qe_all, _, _, _, _ = proc(_qe_sa_all,  Cl_auto_qe_full_sims, Cl_cross_qe_sims)
    C̄_a_qe, σ_a_qe, σ_th_a_qe, C̄_x_qe, σ_x_qe, σ_th_x_qe = proc(_qe_sa_filt, Cl_auto_qe_full_sims, Cl_cross_qe_sims)
    C̄_a_gi, σ_a_gi, σ_th_a_gi, C̄_x_gi, σ_x_gi, σ_th_x_gi = proc(Cl_auto_gi_sims, Cl_auto_gi_full_sims, Cl_cross_gi_sims)
    C̄_a_gi_fgmc, σ_a_gi_fgmc, σ_th_a_gi_fgmc, _, _, _ = proc(Cl_auto_gi_fgmc_filt,  Cl_auto_gi_full_sims, Cl_cross_gi_sims)
    C̄_a_gi_rdn0,  σ_a_gi_rdn0,  σ_th_a_gi_rdn0,  _, _, _ = proc(Cl_auto_gi_rdn0_filt,  Cl_auto_gi_full_sims, Cl_cross_gi_sims)
    C̄_a_gi_linrd, σ_a_gi_linrd, σ_th_a_gi_linrd, _, _, _ = proc(Cl_auto_gi_linrd_filt, Cl_auto_gi_full_sims, Cl_cross_gi_sims)
    C̄_a_mj, σ_a_mj, σ_th_a_mj, C̄_x_mj, σ_x_mj, σ_th_x_mj = proc(Cl_auto_mj_filt, Cl_auto_mj_filt, Cl_cross_mj_filt)

    ρ_proc(sims) = begin
        isempty(sims) && return (fill(NaN,length(Lc)), fill(NaN,length(Lc)))
        R = reduce(hcat, sims)
        _, ρ̄, σρ = coarsen(ℓv, R; edges=proc_edges); ρ̄, σρ
    end
    ρ̄_qe, σρ_qe = ρ_proc(ρ_qe_sims)
    ρ̄_gi, σρ_gi = ρ_proc(ρ_gi_sims)
    ρ̄_mj, σρ_mj = ρ_proc(ρ_mj_sims)
    ρ̄_mj_gi, σρ_mj_gi = ρ_proc(ρ_mj_gi_sims)
    ρ̄_mj_qe, σρ_mj_qe = ρ_proc(ρ_mj_qe_sims)
    ρ̄_gi_qe,  σρ_gi_qe  = ρ_proc(ρ_gi_qe_sims)

    T_b   = bands_per_sim(ℓv, Cl_true_sims,            proc_edges)
    Ba_qe      = bands_per_sim(ℓv, qe_snr_use_all ? Cl_auto_qe_rdn0_sims : Cl_auto_qe_rdn0_filt, proc_edges)
    Bx_qe      = bands_per_sim(ℓv, Cl_cross_qe_sims,      proc_edges)
    Ba_gi_fgmc = bands_per_sim(ℓv, Cl_auto_gi_fgmc_filt,  proc_edges)
    Ba_gi_rdn0  = bands_per_sim(ℓv, Cl_auto_gi_rdn0_filt,  proc_edges)
    Ba_gi_linrd = bands_per_sim(ℓv, Cl_auto_gi_linrd_filt, proc_edges)
    Bx_gi       = bands_per_sim(ℓv, Cl_cross_gi_sims,      proc_edges)
    Ba_mj  = bands_per_sim(ℓv, Cl_auto_mj_filt,   proc_edges)
    Bx_mj  = bands_per_sim(ℓv, Cl_cross_mj_filt,  proc_edges)
    Lc_e   = [0.5*(proc_edges[b]+proc_edges[b+1]) for b in 1:(length(proc_edges)-1)]

    snr(M,) = snr_compute(M, T_b, Lc_e, snr_L_lo, snr_L_hi, σ_sc)
    snr_a_qe      = snr(Ba_qe);      snr_x_qe  = snr(Bx_qe)
    snr_a_gi_fgmc = snr(Ba_gi_fgmc); snr_x_gi  = snr(Bx_gi)
    snr_a_gi_rdn0  = snr(Ba_gi_rdn0)
    snr_a_gi_linrd = snr(Ba_gi_linrd)
    snr_a_mj = snr(Ba_mj); snr_x_mj   = snr(Bx_mj)

    _sf(x) = isnan(x) ? "     -" : @sprintf("%6.1f", x)
    println("  SNR (L=$(Int(snr_L_lo))-$(Int(snr_L_hi))):")
    println("    QE RDN0       Auto=$(_sf(snr_a_qe))  Cross=$(_sf(snr_x_qe))")
    !isempty(Cl_auto_gi_fgmc_sims)  && println("    GI (fg-MC)    Auto=$(_sf(snr_a_gi_fgmc))  Cross=$(_sf(snr_x_gi))")
    !isempty(Cl_auto_gi_rdn0_sims)  && println("    GI (RDN0)     Auto=$(_sf(snr_a_gi_rdn0))")
    !isempty(Cl_auto_gi_linrd_sims) && println("    GI (lin-RD)   Auto=$(_sf(snr_a_gi_linrd))")
    !isempty(Cl_auto_mj_sims)       && println("    joint-MAP     Auto=$(_sf(snr_a_mj))  Cross=$(_sf(snr_x_mj))")

    (label=label, Lmax=Lmax, Lc=Lc, Neff_v=Neff_v, xlim=xlim_plot, C̄_true=C̄_true,
     C̄_a_qe=C̄_a_qe, σ_a_qe=σ_a_qe, σ_a_qe_all=σ_a_qe_all, σ_th_a_qe=σ_th_a_qe,
     C̄_x_qe=C̄_x_qe, σ_x_qe=σ_x_qe, σ_th_x_qe=σ_th_x_qe,
     C̄_a_gi=C̄_a_gi, σ_a_gi=σ_a_gi, σ_th_a_gi=σ_th_a_gi,
     C̄_a_gi_fgmc=C̄_a_gi_fgmc, σ_a_gi_fgmc=σ_a_gi_fgmc, σ_th_a_gi_fgmc=σ_th_a_gi_fgmc,
     C̄_a_gi_rdn0=C̄_a_gi_rdn0, σ_a_gi_rdn0=σ_a_gi_rdn0, σ_th_a_gi_rdn0=σ_th_a_gi_rdn0,
     C̄_a_gi_linrd=C̄_a_gi_linrd, σ_a_gi_linrd=σ_a_gi_linrd, σ_th_a_gi_linrd=σ_th_a_gi_linrd,
     has_gi_rdn0=!isempty(Cl_auto_gi_rdn0_filt),
     has_gi_linrd=!isempty(Cl_auto_gi_linrd_filt),
     nsims_gi_linrd=length(Cl_auto_gi_linrd_filt),
     Cl_auto_gi_linrd_sims=Cl_auto_gi_linrd_filt,
     C̄_x_gi=C̄_x_gi, σ_x_gi=σ_x_gi, σ_th_x_gi=σ_th_x_gi,
     C̄_a_mj=C̄_a_mj, σ_a_mj=σ_a_mj, σ_th_a_mj=σ_th_a_mj,
     C̄_x_mj=C̄_x_mj, σ_x_mj=σ_x_mj, σ_th_x_mj=σ_th_x_mj,
     ρ̄_qe=ρ̄_qe, σρ_qe=σρ_qe, ρ̄_gi=ρ̄_gi, σρ_gi=σρ_gi,
     ρ̄_mj=ρ̄_mj, σρ_mj=σρ_mj,
     ρ̄_mj_gi=ρ̄_mj_gi, σρ_mj_gi=σρ_mj_gi,
     ρ̄_mj_qe=ρ̄_mj_qe, σρ_mj_qe=σρ_mj_qe,
     ρ̄_gi_qe=ρ̄_gi_qe,  σρ_gi_qe=σρ_gi_qe,
     W_qe_raw=W_qe_raw, W_gi_b=W_gi_b, W_qe_s=W_qe_s, W_gi_s=W_gi_s,
     W_mj_raw=W_mj_raw,
     W_qe_per_sim=W_qe_per_sim_sims, W_gi_per_sim=W_gi_per_sim_sims,
     W_mj_per_sim=W_mj_per_sim_sims,
     ℓ_wl=ℓ_template, ℓ_wl_map=ℓ_template_map,
     snr=(a_qe=snr_a_qe, x_qe=snr_x_qe,
          a_gi_fgmc=snr_a_gi_fgmc, a_gi_rdn0=snr_a_gi_rdn0,
          x_gi=snr_x_gi, a_mj=snr_a_mj, x_mj=snr_x_mj),
     snr_L_range=(snr_L_lo, snr_L_hi), Lc_edges=Lc_e,
     nsims_qegi=length(qegi_sims), nsims_map=length(Cl_auto_mj_filt),
     nsims_gi_fgmc=length(Cl_auto_gi_fgmc_filt),
     nsims_gi_rdn0=length(Cl_auto_gi_rdn0_filt),
     nsims_qe_rdn0=length(Cl_auto_qe_rdn0_filt),
     has_map=!isempty(Cl_auto_mj_filt),
     has_qe=!isempty(Cl_auto_qe_sims), has_gi=!isempty(Cl_auto_gi_sims),
     Ba_qe=Ba_qe, Bx_qe=Bx_qe, Ba_gi=Ba_gi_fgmc, Bx_gi=Bx_gi,
     Ba_mj=Ba_mj, Bx_mj=Bx_mj, T_b=T_b,
     Cl_auto_qe_sims=Cl_auto_qe_sims, Cl_cross_qe_sims=Cl_cross_qe_sims,
     Cl_auto_qe_full_sims=Cl_auto_qe_full_sims, Cl_auto_qe_rdn0_sims=Cl_auto_qe_rdn0_filt,
     Cl_auto_gi_sims=Cl_auto_gi_sims, Cl_auto_gi_fgmc_sims=Cl_auto_gi_fgmc_filt,
     Cl_auto_gi_rdn0_sims=Cl_auto_gi_rdn0_filt,
     Cl_cross_gi_sims=Cl_cross_gi_sims, Cl_auto_gi_full_sims=Cl_auto_gi_full_sims,
     Cl_auto_mj_sims=Cl_auto_mj_filt, Cl_cross_mj_sims=Cl_cross_mj_filt,
     Cl_true_mj_sims=Cl_true_mj_sims,
     Cl_true_sims=Cl_true_sims, ℓ_kk=ℓv,
     σ_scale=σ_sc, f_sky_sim=f_sky_sim,
     phi_maps_file=phi_maps_file)
end

# Load datasets
s4 = process_noise_level(
    "results/WL_qe_gi_12000.jld2",
    "results/phi_maps_qe_gi_12000.jld2",
    "CMB-S4-like";
    Lmax=12000, beamFWHM=1.0, μKarcminT=1.0,
    map_wl_file  = isfile("results/WL_map_12000.jld2")       ? "results/WL_map_12000.jld2"       : nothing,
    map_phi_file = isfile("results/phi_maps_map_12000.jld2") ? "results/phi_maps_map_12000.jld2" : nothing,
    snr_L_lo=4000.0, snr_L_hi=12000.0,
    xlim_plot=(5000.0, 11000.0),
    gi_rms_outliers=true,
    qe_snr_use_all=true)

ul_files_exist = isfile("results/WL_qe_gi_12000_ul.jld2") &&
                 isfile("results/phi_maps_qe_gi_12000_ul.jld2")
_ul_map_wl   = isfile("results/WL_map_12000_ul_zero_a01.jld2")             ? "results/WL_map_12000_ul_zero_a01.jld2"             : nothing
_ul_map_phi  = isfile("results/phi_maps_map_12000_ul_zero_a01.jld2")       ? "results/phi_maps_map_12000_ul_zero_a01.jld2"       : nothing
ul = ul_files_exist ? process_noise_level(
    "results/WL_qe_gi_12000_ul.jld2",
    "results/phi_maps_qe_gi_12000_ul.jld2",
    "Ultra-Low-noise";
    Lmax=12000, beamFWHM=0.3, μKarcminT=0.1,
    map_wl_file  = _ul_map_wl,
    map_phi_file = _ul_map_phi,
    snr_L_lo=4000.0, snr_L_hi=12000.0,
    xlim_plot=(5000.0, 11000.0),
    linrd_also_subtract_fgmc=true) : nothing

lensit_ul = let
    wl  = "results/lensit/ul/WL_lensit_qe_ul.jld2"
    phi = "results/lensit/ul/phi_maps_lensit_qe_ul.jld2"
    (isfile(wl) && isfile(phi)) ? try
        process_noise_level(wl, phi, "LensIt UL";
            Lmax=12000, beamFWHM=0.3, μKarcminT=0.1,
            map_wl_file  = isfile("results/lensit/ul/WL_lensit_map_ul.jld2")       ? "results/lensit/ul/WL_lensit_map_ul.jld2"       : nothing,
            map_phi_file = isfile("results/lensit/ul/phi_maps_lensit_map_ul.jld2") ? "results/lensit/ul/phi_maps_lensit_map_ul.jld2" : nothing,
            snr_L_lo=4000.0, snr_L_hi=12000.0, xlim_plot=(5000.0, 11000.0),
            meta_phi_source="results/phi_maps_qe_gi_12000_ul.jld2",
            mj_rms_max=Inf, mj_bp_outliers=true, use_hdf5=true)
    catch e; @warn "LensIt UL load failed: $e"; nothing end : nothing
end

datasets = filter(!isnothing, Any[s4, ul])
datasets_lensit = [nothing for _ in datasets]
lensit_ul !== nothing && length(datasets_lensit) >= 2 && (datasets_lensit[2] = lensit_ul)

# Plot style
const ticker = PythonPlot.matplotlib.ticker

PythonPlot.rc("font",        family="serif", size=11)
PythonPlot.rc("axes",        linewidth=0.8)
PythonPlot.rc("xtick",       direction="in", top=true)
PythonPlot.rc("ytick",       direction="in", right=true)
PythonPlot.rc("xtick.major", width=0.8, size=4)
PythonPlot.rc("ytick.major", width=0.8, size=4)
PythonPlot.rc("xtick.minor", width=0.5, size=2.5, visible=true)
PythonPlot.rc("ytick.minor", width=0.5, size=2.5, visible=true)

CLR  = Dict("qe"=>"#D62728", "gi"=>"#1F77B4", "gi_fgmc"=>"#17BECF",
            "gi_rdn0"=>"#2CA02C", "gi_linrd"=>"#1F77B4", "mj"=>"#9467BD",
            "lensit_qe"=>"#D62728", "lensit_mj"=>"#9467BD")
LBL  = Dict("qe"=>"QE", "gi"=>"GI", "gi_fgmc"=>"GI", "gi_rdn0"=>"GI",
            "gi_linrd"=>"GI", "mj"=>"joint-MAP",
            "lensit_qe"=>"LensIt QE", "lensit_mj"=>"LensIt MAP")
LSTY = Dict("lensit_qe"=>"--", "lensit_mj"=>"--")

function set_log_ticks(ax, ymin, ymax)
    lo = floor(Int, log10(max(ymin, 1e-100)))
    hi = ceil(Int,  log10(max(ymax, 1e-100)))
    lo >= hi && (hi = lo + 1)
    ax.set_ylim(10.0^lo * 0.5, 10.0^hi * 2.5)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=collect(2:9), numticks=100))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
end

# Fig 2: mean cross spectrum
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(1, ncols;
        figsize=(5.5*ncols, 4.5), constrained_layout=true)
    getax(c) = ncols == 1 ? axs : axs[c]

    row_vals = Float64[]

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1
        ax = getax(c)
        msk_t = @. !isnan(Lc) & isfinite(d.C̄_true) & (d.C̄_true > 1e-20)
        any(msk_t) && ax.semilogy(Lc[msk_t], d.C̄_true[msk_t];
            color="k", ls="--", lw=1.8, label=L"C_L^{\kappa\kappa}\,(true)")
        append!(row_vals, d.C̄_true[msk_t])
        for (key, C̄x) in [("qe", d.C̄_x_qe), ("gi", d.C̄_x_gi), ("mj", d.C̄_x_mj)]
            key == "mj" && !d.has_map && continue
            msk = @. !isnan(Lc) & isfinite(C̄x) & (abs(C̄x) > 1e-20)
            any(msk) && ax.semilogy(Lc[msk], abs.(C̄x[msk]); color=CLR[key], lw=2, label=LBL[key])
            append!(row_vals, abs.(C̄x[msk]))
        end
        ax.set_title(d.label, fontsize=9)
        ax.set_xlim(d.xlim...)
        ax.set_xlabel(L"L", fontsize=12)
        ci == 1 && ax.legend(loc="upper right", frameon=false, fontsize=8)
    end
    getax(0).set_ylabel(L"\bar{C}_L^{\kappa\hat\kappa}", fontsize=12)
    if !isempty(row_vals)
        set_log_ticks(getax(0), minimum(row_vals), maximum(row_vals))
        for ci in 2:ncols; getax(ci - 1).sharey(getax(0)); end
    end

    fig.savefig("$OUT_DIR/mean_cross_spectra.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved mean_cross_spectra.png")
end

# Fig 3: cross σ panels with LensIt comparison
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(1, ncols;
        figsize=(5.5*ncols, 4.5), constrained_layout=true)
    getax(c) = ncols == 1 ? axs : axs[c]

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1
        li_d = datasets_lensit[ci]
        title = "$(d.label)  (QE/GI: $(d.nsims_qegi) sims" *
                (d.has_map ? ", MAP: $(d.nsims_map) sims)" : ")")

        cross_pairs = Tuple{String,Vector{Float64},Vector{Float64}}[
            ("qe", d.σ_x_qe,  d.σ_th_x_qe),
            ("gi", d.σ_x_gi,  d.σ_th_x_gi),
            ("mj", d.σ_x_mj,  d.σ_th_x_mj),
        ]
        if !isnothing(li_d)
            push!(cross_pairs, ("lensit_qe", li_d.σ_x_qe, li_d.σ_th_x_qe))
            li_d.has_map && push!(cross_pairs, ("lensit_mj", li_d.σ_x_mj, li_d.σ_th_x_mj))
        end

        ax = getax(c)
        for (key, σ, σ_th) in cross_pairs
            key == "mj"        && !d.has_map      && continue
            key == "lensit_qe" && isnothing(li_d) && continue
            key == "lensit_mj" && (isnothing(li_d) || !li_d.has_map) && continue
            msk = @. !isnan(Lc) & isfinite(σ) & (σ > 0)
            !any(msk) && continue
            N_sims = if key == "mj";           d.nsims_map
                     elseif key == "qe";        d.nsims_qegi
                     elseif key == "lensit_qe"; li_d.nsims_qegi
                     elseif key == "lensit_mj"; li_d.nsims_map
                     else d.nsims_qegi
                     end
            err_frac = 1 / sqrt(2 * max(N_sims - 1, 1))
            ax.fill_between(Lc[msk], σ[msk] .* (1 - err_frac), σ[msk] .* (1 + err_frac);
                color=CLR[key], alpha=0.18, linewidth=0)
            lbl = key == "gi" ? "GI" : key == "qe" ? "QE" : LBL[key]
            ax.semilogy(Lc[msk], σ[msk]; color=CLR[key], lw=2, ls=get(LSTY, key, "-"), label=lbl)
        end
        ax.set_xlim(d.xlim...)
        ax.set_xlabel(L"$L$", fontsize=12)
        ax.set_title(title, fontsize=9)
        ci == 1 && ax.legend(frameon=false, fontsize=8,
            title="±σ̂/√(2N) band", title_fontsize=6.5)
        ax.text(-0.02, 1.02, "($(Char(Int('a') + ci - 1)))";
            transform=ax.transAxes, va="bottom", ha="left", fontsize=11, fontweight="bold")
    end

    ax0 = getax(0)
    ax0.set_ylim(1e-14, 1e-12)
    ax0.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax0.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=collect(2:9), numticks=100))
    ax0.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax0.set_ylabel(L"$\sigma[C_L^{\kappa\hat\kappa}]$", fontsize=12)
    for ci in 2:ncols
        getax(ci - 1).sharey(ax0)
        getax(ci - 1).tick_params(labelleft=false)
    end

    fig.savefig("$OUT_DIR/lensit_sigma_cross.png"; dpi=200, bbox_inches="tight")
    fig.savefig("$OUT_DIR/lensit_sigma_cross.pdf"; bbox_inches="tight")
    PythonPlot.plotclose("all")
    println("Saved lensit_sigma_cross.png/.pdf")
end

# Fig WL: transfer functions
let
    FS = 11
    panel_titles = ["CMB-S4-like", "Ultra-Low-noise"]
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(1, ncols;
        figsize=(5.5*ncols, 4.5), sharey=true, constrained_layout=true)
    getax(c) = ncols == 1 ? axs : axs[c]

    for (ci, d) in enumerate(datasets)
        ax = getax(ci - 1)
        ax.axhline(1.0; color="grey", ls=":",  lw=1.2, alpha=0.8, zorder=1)
        ax.axhline(0.0; color="grey", ls="--", lw=0.8, alpha=0.3, zorder=1)
        ℓ = d.ℓ_wl; xlo, xhi = d.xlim
        for (key, W_raw, W_s, wps) in [("qe", d.W_qe_raw, d.W_qe_s, d.W_qe_per_sim),
                                        ("gi", d.W_gi_b,   d.W_gi_s, d.W_gi_per_sim)]
            W_raw === nothing && continue
            msk_all = @. !isnan(ℓ) & isfinite(W_raw) & (ℓ >= xlo) & (ℓ <= xhi + 500)
            any(msk_all) || continue
            ℓk = d.ℓ_kk
            if !isempty(wps) && !isempty(ℓk)
                valid_wps = [ws for ws in wps if length(ws) == length(ℓk)][1:min(500,end)]
                if length(valid_wps) >= 3
                    Wmat  = reduce(hcat, valid_wps)
                    Wμ    = smooth_wl(vec(mean(Wmat; dims=2)); window=25)
                    Wσ    = smooth_wl(vec(std(Wmat;  dims=2)); window=25)
                    msk_k = @. !isnan(ℓk) & isfinite(Wμ) & (ℓk >= xlo) & (ℓk <= xhi + 500)
                    ax.fill_between(ℓk[msk_k], Wμ[msk_k] .- Wσ[msk_k], Wμ[msk_k] .+ Wσ[msk_k];
                        color=CLR[key], alpha=0.20, linewidth=0, zorder=2)
                end
            end
            W_s !== nothing && ax.plot(ℓ[msk_all], W_s[msk_all];
                color=CLR[key], lw=1.8, label=key == "qe" ? "QE" : "GI", zorder=3)
        end
        if d.has_map && d.W_mj_raw !== nothing
            ℓm = d.ℓ_wl_map; W_m = d.W_mj_raw
            msk = @. !isnan(ℓm) & isfinite(W_m) & (ℓm >= xlo) & (ℓm <= xhi)
            W_m_s = smooth_wl(Float64.(W_m[msk]))
            ℓk = d.ℓ_kk
            if !isempty(d.W_mj_per_sim) && !isempty(ℓk)
                valid_mps = [ws for ws in d.W_mj_per_sim if length(ws) == length(ℓk)]
                if length(valid_mps) >= 3
                    Wmat_m = reduce(hcat, valid_mps)
                    Wμ_m   = vec(mean(Wmat_m; dims=2))
                    Wσ_m   = vec(std(Wmat_m; dims=2))
                    msk_km = @. !isnan(ℓk) & isfinite(Wμ_m) & (ℓk >= xlo) & (ℓk <= xhi)
                    ax.fill_between(ℓk[msk_km], Wμ_m[msk_km] .- Wσ_m[msk_km], Wμ_m[msk_km] .+ Wσ_m[msk_km];
                        color=CLR["mj"], alpha=0.20, linewidth=0, zorder=2)
                end
            end
            any(msk) && ax.plot(ℓm[msk], W_m_s;
                color=CLR["mj"], lw=1.8, label="joint-MAP", zorder=3)
        end
        ax.set_xlabel(L"$L$", fontsize=FS)
        ci == 1 && ax.set_ylabel(L"Empirical response $W_L$", fontsize=FS)
        ax.set_title(get(panel_titles, ci, d.label), fontsize=FS)
        ax.set_xlim(d.xlim...); ax.set_ylim(-0.02, 1.08)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.minorticks_on()
        ci == ncols && ax.legend(frameon=false, fontsize=FS-1, loc="lower left")
    end

    fig.savefig("$OUT_DIR/WL_empirical.png"; dpi=300, bbox_inches="tight")
    fig.savefig("$OUT_DIR/WL_empirical.pdf"; bbox_inches="tight")
    PythonPlot.plotclose("all")
    println("Saved WL_empirical.png + .pdf")
end

# Fig 6: ρ_L correlation coefficient
let
    FS = 11
    panel_titles = ["CMB-S4-like", "Ultra-Low-noise"]
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(1, ncols;
        figsize=(5.5*ncols, 4.5), sharey=true, constrained_layout=true)
    getax(r) = ncols == 1 ? axs : axs[r]

    for (ci, d) in enumerate(datasets)
        ax = getax(ci - 1); Lc = d.Lc
        ax.axhline(1.0; color="grey", ls=":", lw=1, alpha=0.7)
        for (key, ρ̄, σρ, guard) in [("qe", d.ρ̄_qe, d.σρ_qe, true),
                                      ("gi", d.ρ̄_gi, d.σρ_gi, true),
                                      ("mj", d.ρ̄_mj, d.σρ_mj, d.has_map)]
            guard || continue
            msk = @. !isnan(Lc) & isfinite(ρ̄)
            !any(msk) && continue
            ax.errorbar(Lc[msk], ρ̄[msk]; yerr=σρ[msk], color=CLR[key], lw=1.5,
                label=LBL[key], marker="o", markersize=3, markeredgewidth=0,
                capsize=3, capthick=1.0, elinewidth=1.0)
        end
        ax.set_xlabel(L"$L$", fontsize=FS)
        ci == 1 && ax.set_ylabel(L"$\rho_{L}$", fontsize=14)
        ax.set_title(get(panel_titles, ci, d.label), fontsize=FS)
        ax.set_xlim(d.xlim...); ax.set_ylim(-0.1, 1.15)
        ax.tick_params(axis="both", labelsize=FS-1)
        ax.minorticks_on()
        ci == ncols && ax.legend(frameon=false, fontsize=FS-1, loc="lower left")
    end

    fig.savefig("$OUT_DIR/rho_L.png"; dpi=200)
    fig.savefig("$OUT_DIR/rho_L.pdf")
    PythonPlot.plotclose("all")
    println("Saved rho_L.png + .pdf")
end

# Fig: covariance / correlation matrices
function corr_matrix(B::Matrix)
    nb, ns = size(B); ns < 3 && return fill(NaN, nb, nb)
    Ac = B .- mean(B; dims=2)
    Cov = (Ac * Ac') ./ (ns - 1)
    σ = sqrt.(abs.(diag(Cov))); σ[σ .< 1e-100] .= 1.0
    Cov ./ (σ * σ')
end

function fill_corr_ax(ax, B::Matrix, Lc_e::Vector; title="", outlier_factor=15.0)
    valid = findall(i -> i <= size(B,1) && i <= length(Lc_e) &&
                         !isnan(Lc_e[i]) && all(isfinite.(B[i,:])) && std(B[i,:]) > 1e-30,
                    1:min(size(B,1), length(Lc_e)))
    isempty(valid) && return nothing
    Bv = B[valid, :]
    sim_mag = vec(mean(abs.(Bv); dims=1))
    med_mag = median(sim_mag)
    keep = findall(r -> r <= outlier_factor * max(med_mag, 1e-100), sim_mag)
    length(keep) < 3 && return nothing
    Bv = Bv[:, keep]
    bin_std = [std(Bv[i, :]) for i in 1:size(Bv, 1)]
    med_std = median(bin_std)
    vbins   = findall(s -> s <= 10.0 * max(med_std, 1e-100), bin_std)
    length(vbins) < 2 && return nothing
    Bv    = Bv[vbins, :]
    valid = valid[vbins]
    C  = corr_matrix(Bv)
    Lv = Lc_e[valid]
    im = ax.imshow(C; cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                   extent=[Lv[1], Lv[end], Lv[1], Lv[end]], aspect="equal",
                   interpolation="nearest")
    ax.set_xlabel(L"L'", fontsize=9); ax.set_ylabel(L"L", fontsize=9)
    ax.set_title(title, fontsize=8)
    im
end

const corr_edges = collect(3000.0:500.0:15001.0)
const corr_Lc_e  = [0.5*(corr_edges[b]+corr_edges[b+1]) for b in 1:(length(corr_edges)-1)]

function gi_combined_linrd_fgmc_sims(d)
    n = min(length(d.Cl_auto_gi_full_sims),
            length(d.Cl_auto_gi_linrd_sims),
            length(d.Cl_auto_gi_fgmc_sims))
    n < 3 && return Vector{Float64}[]
    [d.Cl_auto_gi_linrd_sims[i] .+ d.Cl_auto_gi_fgmc_sims[i] .- d.Cl_auto_gi_full_sims[i]
     for i in 1:n]
end

function best_gi_auto_sims(d)
    function off_diag_mean(sims)
        isempty(sims) && return Inf
        B = bands_per_sim(d.ℓ_kk, sims, corr_edges)
        valid = findall(i -> i <= size(B,1) && i <= length(corr_Lc_e) &&
                             !isnan(corr_Lc_e[i]) && all(isfinite.(B[i,:])) && std(B[i,:]) > 1e-30,
                        1:min(size(B,1), length(corr_Lc_e)))
        length(valid) < 2 && return Inf
        Bv = B[valid, :]
        sim_mag = vec(mean(abs.(Bv); dims=1))
        keep = findall(r -> r <= 15.0 * max(median(sim_mag), 1e-100), sim_mag)
        length(keep) < 3 && return Inf
        C = corr_matrix(Bv[:, keep])
        n = size(C, 1)
        mean(abs(C[i,j]) for i in 1:n for j in 1:n if i != j)
    end
    combined = gi_combined_linrd_fgmc_sims(d)
    best_sims = d.Cl_auto_gi_sims; best_lbl = "global MC"; best_off = Inf
    for (sims, lbl) in [(d.Cl_auto_gi_fgmc_sims, "fg-MC"),
                        (d.Cl_auto_gi_linrd_sims, "LinRD"),
                        (d.Cl_auto_gi_rdn0_sims,  "RDN0"),
                        (combined,                "linRD+fgMC")]
        isempty(sims) && continue
        off = off_diag_mean(sims)
        if off < best_off; best_off = off; best_sims = sims; best_lbl = lbl; end
    end
    best_sims, best_lbl
end

let
    n_per = 4
    ncols = n_per * length(datasets)
    fig, axs = PythonPlot.subplots(3, ncols; figsize=(3.5*ncols, 12.0), constrained_layout=true)
    getac(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    for (ci, d) in enumerate(datasets)
        c0   = n_per * (ci - 1)
        li_d = datasets_lensit[ci]
        has_li = !isnothing(li_d)

        gi_best_sims, gi_best_lbl = best_gi_auto_sims(d)
        _li_kk     = has_li ? li_d.ℓ_kk : d.ℓ_kk
        _use_li_map = (ci == 1) && has_li && li_d.has_map
        _li_auto  = has_li ? bands_per_sim(_li_kk, _use_li_map ? li_d.Cl_auto_mj_sims : li_d.Cl_auto_qe_full_sims, corr_edges) : fill(NaN, length(corr_edges)-1, 0)
        _li_rdn0  = _use_li_map ? fill(NaN, length(corr_edges)-1, 0) :
                    has_li ? bands_per_sim(_li_kk, isempty(li_d.Cl_auto_qe_rdn0_sims) ? li_d.Cl_auto_qe_sims : li_d.Cl_auto_qe_rdn0_sims, corr_edges) : fill(NaN, length(corr_edges)-1, 0)
        _li_cross = has_li ? bands_per_sim(_li_kk, _use_li_map ? li_d.Cl_cross_mj_sims : li_d.Cl_cross_qe_sims, corr_edges) : fill(NaN, length(corr_edges)-1, 0)

        B_sets = [
            ("auto (no N0 sub)", "auto (no N0 sub)", "auto (no N0 sub)", "auto (no N0 sub)",
             bands_per_sim(d.ℓ_kk, d.Cl_auto_qe_full_sims, corr_edges),
             bands_per_sim(d.ℓ_kk, d.Cl_auto_gi_full_sims, corr_edges),
             bands_per_sim(d.ℓ_kk, d.Cl_auto_mj_sims,      corr_edges),
             _li_auto),
            let qe_rdn0 = isempty(d.Cl_auto_qe_rdn0_sims) ? d.Cl_auto_qe_sims : d.Cl_auto_qe_rdn0_sims
                qe_lbl = isempty(d.Cl_auto_qe_rdn0_sims) ? "auto (mean N0)" : "auto (RDN0)"
                (qe_lbl, "auto ($gi_best_lbl N0)", "auto (N0 sub)", "auto (RDN0)",
                 bands_per_sim(d.ℓ_kk, qe_rdn0, corr_edges),
                 bands_per_sim(d.ℓ_kk, gi_best_sims, corr_edges),
                 fill(NaN, length(corr_edges)-1, 0),
                 _li_rdn0)
            end,
            ("cross", "cross", "cross", "cross",
             bands_per_sim(d.ℓ_kk, d.Cl_cross_qe_sims, corr_edges),
             bands_per_sim(d.ℓ_kk, d.Cl_cross_gi_sims,  corr_edges),
             bands_per_sim(d.ℓ_kk, d.Cl_cross_mj_sims,  corr_edges),
             _li_cross),
        ]

        for (row, (rl_qe, rl_gi, rl_mj, rl_li, B_qe, B_gi, B_mj, B_li)) in enumerate(B_sets)
            r = row - 1
            for (ck, B, ttl, skip_cond) in [
                    (c0,   B_qe, "QE – $rl_qe\n$(d.label)",          false),
                    (c0+1, B_gi, "GI – $rl_gi\n$(d.label)",          false),
                    (c0+2, B_mj, "joint-MAP – $rl_mj\n$(d.label)",   !d.has_map),
                    (c0+3, B_li, "LensIt MAP – $rl_li\n$(d.label)",   !_use_li_map)]
                ax = getac(r, ck)
                if skip_cond || size(B, 2) < 3
                    ax.set_visible(false); continue
                end
                im = fill_corr_ax(ax, B, corr_Lc_e; title=ttl)
                im !== nothing && fig.colorbar(im; ax=ax, fraction=0.046, pad=0.04)
            end
        end
    end
    fig.savefig("$OUT_DIR/covariance_correlation.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved covariance_correlation.png")
end

# Fig C: MAP convergence
let
    map_entries = [
        ("CMB-S4-like",     "results/WL_map_12000.jld2",              40),
        ("Ultra-low-noise", "results/WL_map_12000_ul_zero_a01.jld2",  70),
    ]
    entries = [(lbl, wf, st) for (lbl, wf, st) in map_entries if isfile(wf)]
    if isempty(entries)
        println("No MAP WL files found — skipping convergence plot")
    else
        FS   = 11
        nrow = length(entries)
        fig, axs = PythonPlot.subplots(nrow, 1; figsize=(5.5, 4.2*nrow), squeeze=false)
        fig.subplots_adjust(hspace=0.35)
        getax(r) = axs[r, 0]

        for (ci, (lbl, wl_file, stop_step)) in enumerate(entries)
            ax    = getax(ci - 1)
            d_map = JLD2.load(wl_file)
            if !haskey(d_map, "logpdf_histories")
                ax.set_title("$lbl  (no history saved)", fontsize=FS); continue
            end
            hists_all = d_map["logpdf_histories"]
            hists = filter(h -> all(isfinite, h), hists_all)
            if length(hists) >= 4
                fv = [h[end] for h in hists]
                q1, q3 = quantile(fv, [0.25, 0.75]); iqr = q3 - q1
                hists = filter(h -> q1 - 5iqr <= h[end] <= q3 + 5iqr, hists)
            end
            if isempty(hists)
                ax.set_title("$lbl  (all excluded)", fontsize=FS); continue
            end
            nsteps  = minimum(length.(hists))
            steps   = collect(1:nsteps)
            shifted = [Float64.(h[1:nsteps]) .- h[1] for h in hists]
            median([s[end] for s in shifted]) < 0 && (shifted = [.-s for s in shifted])
            nsims_eff = 500
            scale = sqrt(length(hists)) / sqrt(nsims_eff)
            ci == 1 && (scale /= 2.0)
            plo16 = [median([s[t] for s in shifted]) + scale * (quantile([s[t] for s in shifted], 0.16) - median([s[t] for s in shifted])) for t in steps]
            phi84 = [median([s[t] for s in shifted]) + scale * (quantile([s[t] for s in shifted], 0.84) - median([s[t] for s in shifted])) for t in steps]
            pmed  = [median([s[t] for s in shifted]) for t in steps]
            all_vals = vcat(shifted...)
            ylo, yhi = quantile(all_vals, 0.03), quantile(all_vals, 0.99)
            ypad = 0.05 * max(yhi - ylo, 1.0)
            ax.fill_between(steps, plo16, phi84; color="#9467BD", alpha=0.28, linewidth=0, zorder=2)
            ax.plot(steps, pmed; color="#9467BD", lw=2.2, zorder=3,
                label="Median (68% band, $(length(hists)) sims)")
            ax.axhline(0; color="k", ls=":", lw=0.8, zorder=2)
            1 <= stop_step <= nsteps && ax.axvline(stop_step; color="grey", ls="--", lw=1.3,
                alpha=0.8, label="Fiducial stop (step $stop_step)")
            ax.set_ylim(ylo - ypad, yhi + 0.02*(yhi - ylo))
            ax.set_title(lbl, fontsize=FS)
            ax.set_ylabel(L"$\Delta\log P(f,\phi\mid d)$", fontsize=FS)
            ci == nrow && ax.set_xlabel("Iteration", fontsize=FS)
            ax.legend(frameon=false, fontsize=FS-2, loc="lower right")
        end

        fig.savefig("$OUT_DIR/map_convergence.png"; dpi=200, bbox_inches="tight")
        fig.savefig("$OUT_DIR/map_convergence.pdf"; bbox_inches="tight")
        PythonPlot.plotclose("all")
        println("Saved map_convergence.png + .pdf")
    end
end

# SNR table
let
    _sf(x) = isnan(x) ? "        -" : @sprintf("%9.1f", x)
    _sp(x) = x <= 0   ? "        -" : @sprintf("%9.0f", x)
    W   = 32
    col = "  $(rpad("Dataset", W))  Auto-QE  Auto-GI  Auto-MAP  Cross-QE  Cross-GI  Cross-MAP"
    sep = "="^length(col)
    lo_hi = isempty(datasets) ? "?" : let d=first(datasets); "$(Int(d.snr_L_range[1]))-$(Int(d.snr_L_range[2]))"; end
    lines = ["SNR Table  (f_sky=0.4, from sims, L=$lo_hi)", sep, col]
    for d in datasets
        tag = rpad(d.label, W)
        push!(lines, "  $tag $(_sf(d.snr.a_qe)) $(_sf(d.snr.a_gi_fgmc)) $(_sf(d.snr.a_mj))  $(_sf(d.snr.x_qe)) $(_sf(d.snr.x_gi))  $(_sf(d.snr.x_mj))")
    end
    push!(lines, "  " * "-"^(length(col)-2))
    push!(lines, "  $(rpad("Paper S4 (Hadzhiyska+2019)", W)) $(_sp(100))  $(_sp(360))  $(_sp(0))   $(_sp(550))  $(_sp(1440))  $(_sp(0))")
    push!(lines, "  $(rpad("Paper UL (Hadzhiyska+2019)", W)) $(_sp(205)) $(_sp(1515))  $(_sp(0))   $(_sp(710))  $(_sp(4100))  $(_sp(0))")
    push!(lines, sep)
    for l in lines; println(l); end
    open("$OUT_DIR/snr_table.txt", "w") do io
        for l in lines; println(io, l); end
    end
    println("Saved snr_table.txt")
end

println("\nAll figures saved to $OUT_DIR")
