using PythonPlot
using Statistics: quantile
using CMBLensing: Map
plt = PythonPlot

"""
    plot_grad_T_lensed(f̃)

Three-panel plot of ∂T/∂x, ∂T/∂y, and |∇T| for a lensed temperature field.
"""
function plot_grad_T_lensed(f̃)
    dTdx, dTdy, grad_mag = grad_fft(f̃)

    lo = quantile(vec(grad_mag), 0.2)
    hi = quantile(vec(grad_mag), 0.8)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(dTdx, cmap="RdBu_r", vmin=lo, vmax=hi)
    plt.title(L"\partial T / \partial x")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(dTdy, cmap="RdBu_r", vmin=lo, vmax=hi)
    plt.title(L"\partial T / \partial y")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(grad_mag, cmap="viridis", vmin=lo, vmax=hi)
    plt.title(L"|\nabla T|")
    plt.colorbar()

    plt.tight_layout()
end

"""
    plot_phi_error_vs_gradT(f, mean_Δϕ²_qe, mean_Δϕ²_joint, mean_Δϕ²_marg)

Binned mean |Δϕ|² vs |∇T| for QE, joint MAP, and MAP marg, with 1σ shaded bands.
"""
function plot_phi_error_vs_gradT(f, mean_Δϕ²_qe::Matrix, mean_Δϕ²_joint::Matrix, mean_Δϕ²_marg::Matrix;
                                 nbins=20, min_count=10)
    _, _, grad_mag = grad_fft(f)

    cen, m_QE,   s_QE,   n_QE   = bin_stat_err(Float64.(grad_mag), mean_Δϕ²_qe;   nbins=nbins)
    _,   m_J,    s_J,    n_J    = bin_stat_err(Float64.(grad_mag), mean_Δϕ²_joint; nbins=nbins)
    _,   m_marg, s_marg, n_marg = bin_stat_err(Float64.(grad_mag), mean_Δϕ²_marg;  nbins=nbins)

    ok_QE   = n_QE   .>= min_count
    ok_J    = n_J    .>= min_count
    ok_marg = n_marg .>= min_count

    plt.rc("font",   size=12, family="serif")
    plt.rc("axes",   linewidth=0.8)
    plt.rc("xtick",  direction="in", top=true)
    plt.rc("ytick",  direction="in", right=true)
    plt.rc("xtick.major", width=0.8, size=4)
    plt.rc("ytick.major", width=0.8, size=4)

    colours = ["#4477AA", "#EE6677", "#228833"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5.5, 5.5),
        gridspec_kw=Dict("height_ratios" => [3, 1], "hspace" => 0.08),
        sharex=true)

    # ── Top panel: |Δϕ|² vs |∇T| ──
    for (ok, m, s, col, lab, mk) in [
        (ok_QE,   m_QE,   s_QE,   colours[1], "QE",        "o"),
        (ok_J,    m_J,    s_J,    colours[2], "Joint MAP",  "s"),
        (ok_marg, m_marg, s_marg, colours[3], "MAP marg",   "^"),
    ]
        c, μ, σ = cen[ok], m[ok], s[ok]
        ax_top.plot(c, μ, marker=mk, color=col, label=lab,
                markersize=4, linewidth=1.2, markeredgewidth=0.6, markeredgecolor="white")
        ax_top.fill_between(c, μ .- σ, μ .+ σ, alpha=0.2, color=col)
    end

    ax_top.set_yscale("log")
    ax_top.set_ylabel(L"\langle|\Delta\phi|^2\rangle", fontsize=13)
    ax_top.legend(frameon=false, fontsize=11)
    ax_top.tick_params(labelsize=11, labelbottom=false)
    ax_top.spines["top"].set_visible(false)
    ax_top.spines["right"].set_visible(false)

    # ── Bottom panel: pixel count histogram ──
    grad_vec = vec(Float64.(grad_mag))
    lo, hi = minimum(grad_vec), maximum(grad_vec)
    hist_edges = range(lo, hi; length=nbins + 1)

    ax_bot.hist(grad_vec, bins=collect(hist_edges), color="grey", alpha=0.6, edgecolor="none")
    ax_bot.set_xlabel(L"|\nabla T|\;\mathrm{(per\;pixel)}", fontsize=13)
    ax_bot.set_ylabel(L"N_\mathrm{pixels}", fontsize=11)
    ax_bot.tick_params(labelsize=10)
    ax_bot.spines["top"].set_visible(false)
    ax_bot.spines["right"].set_visible(false)

    fig.tight_layout()
end

"""
    plot_WL_comparison(ℓ_template, WL_joint, WL_marg; WL_analytical=nothing, ℓ_ana=nothing)

Plot empirical W_L for joint/marg, optionally with analytical QE curve.
"""
function plot_WL_comparison(
    ℓ_template, WL_joint, WL_marg, WL_qe;
    WL_analytical=nothing, ℓ_ana=nothing
)
    ℓ  = collect(ℓ_template)
    WJ = collect(WL_joint)
    WM = collect(WL_marg)
    WQ = collect(WL_qe)

    plt.figure(figsize=(7, 5))
    plt.plot(ℓ, WJ, label="MAP joint (empirical)")
    plt.plot(ℓ, WM, label="MAP marg (empirical)")
    plt.plot(ℓ, WQ, label="QE (empirical)")

    if WL_analytical !== nothing
        ℓa = ℓ_ana !== nothing ? collect(ℓ_ana) : ℓ
        plt.plot(ℓa, collect(WL_analytical), label="QE analytical (iterative)",
                 linestyle="--", color="black")
    end

    plt.xlabel(L"\ell")
    plt.ylabel(L"W_\ell")
    plt.xlim(0, 2000)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(true, alpha=0.3)
    plt.tight_layout()
end

"""
    plot_correlation_coefficient(ϕ, ϕJ, ϕ_marg, ds; ΔL=100)

Binned ρ_ℓ for joint MAP, MAP marg, and QE.
"""
function plot_correlation_coefficient(ϕ, ϕJ, ϕ_marg, ds; ΔL=100)
    ρ_map  = CMBLensing.get_ρℓ(ϕ, ϕJ)
    ρ_qe   = CMBLensing.get_ρℓ(quadratic_estimate(ds).ϕqe, ϕ)
    ρ_marg = CMBLensing.get_ρℓ(ϕ_marg, ϕ)

    Lmap,  ρmap_b  = bin_spectrum(ρ_map.ℓ,  ρ_map.Cℓ;  ΔL=ΔL)
    Lqe,   ρqe_b   = bin_spectrum(ρ_qe.ℓ,   ρ_qe.Cℓ;   ΔL=ΔL)
    Lmarg, ρmarg_b = bin_spectrum(ρ_marg.ℓ, ρ_marg.Cℓ; ΔL=ΔL)

    plt.figure(figsize=(7, 4))
    plt.plot(Lmap,  ρmap_b,  label="Joint MAP")
    plt.plot(Lmarg, ρmarg_b, label="MAP marg")
    plt.plot(Lqe,   ρqe_b,   label="QE", linestyle="--")
    plt.axhline(1, color="k", linestyle=":", linewidth=0.5)

    plt.xlim(0, 1200)
    plt.ylim(0, 1.05)
    plt.xlabel(L"\ell")
    plt.ylabel(L"\rho_\ell")
    plt.legend()
    plt.grid(true, alpha=0.2)
    plt.tight_layout()
end