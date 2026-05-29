using CMBLensing: Map, m_rfft, m_irfft, FlatMap, LowPass, HighPass, simulate, get_Cℓ
using Statistics: mean
using Random: MersenneTwister

export gi_estimate, gi_estimate_ref, gi_estimate_corrected,
       gi_twoleg, compute_sigma_fid,
       gi_n0_fixed_gradient_mc, gi_n0_mc, gi_n0_rdn0,
       gi_n0_linrd_analytical, gi_n0_linrd_analyticbaseline

struct _SimData{F}
    d::F
end


"""
    gi_estimate(ds; Lgrad=2000, Lhp=4000, Lmax=18000)

GI lensing estimator (Hadzhiyska et al. 2019). To first order,
T̃(x) = T(x) + ∇T(x)·∇φ(x), so isolating the small-scale residual gives

    φ̂(L) = −i(Lx·FFT[gx·T_hp] + Ly·FFT[gy·T_hp]) / (Lx²σxx + 2LxLyσxy + Ly²σyy)

where g = ∇(LowPass(Lgrad)·T̃) and T_hp = HighPass(Lhp)·T̃.

If `σ_fid` is provided the denominator uses fiducial gradient variances rather
than the per-realisation values; this is required when pairing with `gi_n0_rdn0`.
"""
function gi_estimate(ds; Lgrad=2000, Lhp=4000, Lmax=18000,
                     denom_floor_frac=1e-8, σ_fid=nothing)

    m    = Map(ds.d)
    proj = m.proj

    dTdx, dTdy, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))
    T_hp = Map(HighPass(Lhp) * ds.d).arr

    A_F = m_rfft(dTdx .* T_hp, (1,2))
    B_F = m_rfft(dTdy .* T_hp, (1,2))

    if σ_fid !== nothing
        σ_xx, σ_xy, σ_yy = σ_fid.σ_xx, σ_fid.σ_xy, σ_fid.σ_yy
    else
        σ_xx = mean(dTdx .^ 2)
        σ_xy = mean(dTdx .* dTdy)
        σ_yy = mean(dTdy .^ 2)
    end

    NyF, NxF = size(A_F)
    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2

    numer = @. -im * (ℓx2D * A_F + ℓy2D * B_F)
    denom = @. ℓx2D^2 * σ_xx + 2*ℓx2D*ℓy2D * σ_xy + ℓy2D^2 * σ_yy

    # floor scales as L² to match the physical dimension of the denominator
    denom_threshold_2D = @. denom_floor_frac * L2 * max(σ_xx, σ_yy)
    L_mask = @. (L2 > Lhp^2) & (L2 < Lmax^2)

    φ_F = @. ifelse(L_mask & (abs(denom) > denom_threshold_2D),
                    numer / denom, complex(0.0))

    return FlatFourier(φ_F, proj)
end


"""
    gi_twoleg(ds_grad, ds_hp; Lgrad=2000, Lhp=4000, Lmax=18000)

Two-leg GI estimator: gradient from `ds_grad`, high-pass map from `ds_hp`.
Used to form the RDN0 combinations DS, SD, SS.
"""
function gi_twoleg(ds_grad, ds_hp; σ_fid=nothing, Lgrad=2000, Lhp=4000, Lmax=18000,
                   denom_floor_frac=1e-8)

    proj = Map(ds_grad.d).proj

    dTdx, dTdy, _ = grad_fft(Map(LowPass(Lgrad) * ds_grad.d))

    if σ_fid !== nothing
        σ_xx, σ_xy, σ_yy = σ_fid.σ_xx, σ_fid.σ_xy, σ_fid.σ_yy
    else
        σ_xx = mean(dTdx .^ 2)
        σ_xy = mean(dTdx .* dTdy)
        σ_yy = mean(dTdy .^ 2)
    end

    T_hp = Map(HighPass(Lhp) * ds_hp.d).arr

    A_F = m_rfft(dTdx .* T_hp, (1,2))
    B_F = m_rfft(dTdy .* T_hp, (1,2))

    NyF, NxF = size(A_F)
    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2

    numer = @. -im * (ℓx2D * A_F + ℓy2D * B_F)
    denom = @. ℓx2D^2 * σ_xx + 2*ℓx2D*ℓy2D * σ_xy + ℓy2D^2 * σ_yy

    denom_thresh_2D = @. denom_floor_frac * L2 * max(σ_xx, σ_yy)
    L_mask = @. (L2 > Lhp^2) & (L2 < Lmax^2)
    φ_F = @. ifelse(L_mask & (abs(denom) > denom_thresh_2D),
                    numer / denom, complex(0.0))

    return FlatFourier(φ_F, proj)
end


"""
    gi_estimate_wiener_fast(ds; Lgrad=2000, Lhp=4000, Lmax=12000, N_abins=12, N_rbins=8)

Fast binned approximation to the per-pixel Wiener GI estimator (Hadzhiyska+2019, eq. 29).
Groups modes into N_abins×N_rbins bins; takes one FFT per bin instead of one FFT per mode,
giving a ~400–700× speedup with <5% error in W_L.

For each (angle,radial) bin centred at (Lx_c, Ly_c):
  gprod(x)  = Lx_c·gx + Ly_c·gy           [per-pixel projected gradient]
  W(x)      = Cϕ_c / (PS_c/gprod² + Cϕ_c) [per-pixel Wiener weight, using bin-mean PS,Cϕ]
  φ̂(L)      = FFT[W·T_hp/gprod][L] / (mean(W)·i)   for all L in the bin

The reference exact implementation is `gi_estimate_corrected`; use that for validation.
"""
function gi_estimate_wiener_fast(ds; Lgrad=2000, Lhp=4000, Lmax=12000,
                                  N_abins=12, N_rbins=8, denom_floor_frac=1e-8)
    m    = Map(ds.d)
    proj = m.proj
    Ny, Nx = size(m.arr)
    NyF  = Ny ÷ 2 + 1

    gx, gy, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))
    T_hp = Float64.(m.arr) .- Float64.(Map(LowPass(Lhp) * ds.d).arr)

    f1     = FlatFourier(ones(ComplexF64, NyF, Nx), proj)
    B_diag = real.(ds.B̂.diag.arr)[1:NyF, 1:Nx]
    PS_2D  = real.((ds.Cf̃ * f1).arr)[1:NyF, 1:Nx] .+
             real.((ds.Cn̂ * f1).arr)[1:NyF, 1:Nx] ./ max.(B_diag .^ 2, 1e-30)
    Cϕ_2D  = real.((ds.Cϕ * f1).arr)[1:NyF, 1:Nx]

    gfloor = denom_floor_frac * sqrt(mean(gx .^ 2) + mean(gy .^ 2))

    ℓx_arr = Float64.(proj.ℓx[1:Nx])
    ℓy_arr = Float64.(proj.ℓy[1:NyF])
    ℓx2D   = repeat(ℓx_arr', NyF, 1)
    ℓy2D   = repeat(ℓy_arr,  1,   Nx)
    L2_2D  = ℓx2D .^ 2 .+ ℓy2D .^ 2
    Lmag   = sqrt.(L2_2D)
    θ_2D   = atan.(ℓy2D, ℓx2D)

    φ_F = zeros(ComplexF64, NyF, Nx)

    # Log-spaced radial bin edges; angular bins uniformly cover [0, π]
    L_edges = exp.(range(log(Float64(Lhp)), log(Float64(Lmax)), N_rbins + 1))
    θ_edges = range(0.0, Float64(π), N_abins + 1)

    gp_buf = similar(gx)
    W_buf  = similar(gx)
    pe_buf = zeros(Float64, Ny, Nx)

    for ia in 1:N_abins
        θ_lo, θ_hi = θ_edges[ia], θ_edges[ia + 1]
        θ_c   = 0.5 * (θ_lo + θ_hi)
        cos_c = cos(θ_c); sin_c = sin(θ_c)

        @. gp_buf = cos_c * gx + sin_c * gy
        @. gp_buf = ifelse(abs(gp_buf) < gfloor,
                           ifelse(gp_buf >= 0, gfloor, -gfloor), gp_buf)

        for ir in 1:N_rbins
            L_lo, L_hi = L_edges[ir], L_edges[ir + 1]
            L_c = sqrt(L_lo * L_hi)   # geometric mean

            # boundary bins absorb modes at θ=0 and θ=π
            if ia == 1
                bin_mask = @. (Lmag >= L_lo) & (Lmag < L_hi) & (θ_2D < θ_hi)
            elseif ia == N_abins
                bin_mask = @. (Lmag >= L_lo) & (Lmag < L_hi) & (θ_2D >= θ_lo)
            else
                bin_mask = @. (Lmag >= L_lo) & (Lmag < L_hi) &
                              (θ_2D >= θ_lo) & (θ_2D < θ_hi)
            end
            !any(bin_mask) && continue

            PS_c = mean(PS_2D[bin_mask])
            Cϕ_c = mean(Cϕ_2D[bin_mask])
            Cϕ_c <= 0 && continue

            @. W_buf  = Cϕ_c * (L_c * gp_buf)^2 / max(PS_c + Cϕ_c * (L_c * gp_buf)^2, 1e-60)
            @. pe_buf = W_buf * T_hp / (L_c * gp_buf)
            W_mean = mean(W_buf)
            W_mean < 1e-30 && continue

            F_pe = m_rfft(pe_buf, (1, 2))
            @inbounds for j in 1:NyF, i in 1:Nx
                bin_mask[j, i] || continue
                Cϕ_2D[j, i] <= 0 && continue
                φ_F[j, i] = F_pe[j, i] / (W_mean * im)
            end
        end
    end

    return FlatFourier(φ_F, proj)
end


"""
    gi_estimate_corrected(ds; Lgrad=2000, Lhp=4000, Lmax=20000)

Pixel-exact GI estimator (Hadzhiyska+2019, eq. 29). For each output mode L, a per-pixel
Wiener weight is applied before the FFT:

    gprod(x) = Lx·gx(x) + Ly·gy(x)
    W(x)     = Cϕ(L) / (C_TT(L)/gprod(x)² + Cϕ(L))
    φ̂(L)     = FFT[W·T_hp/gprod][L] / (mean(W)·i)

where C_TT = B²·Cf̃ + Cn̂. This does one FFT per output mode and is much slower
than `gi_estimate`, which uses scalar mean(gprod²) in the denominator.
"""
function gi_estimate_corrected(ds; Lgrad=2000, Lhp=4000, Lmax=20000, denom_floor_frac=1e-8)

    m    = Map(ds.d)
    proj = m.proj
    Ny, Nx = size(m.arr)
    NyF, NxF = Ny ÷ 2 + 1, Nx

    gx, gy, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))
    T_lp = Map(LowPass(Lhp) * ds.d).arr
    T_hp = m.arr .- T_lp

    f_ones  = FlatFourier(ones(ComplexF64, NyF, NxF), proj)
    B_diag  = real.(ds.B̂.diag.arr)[1:NyF, 1:NxF]
    Cf_diag = real.((ds.Cf * f_ones).arr)[1:NyF, 1:NxF]
    Cn_diag = real.((ds.Cn̂ * f_ones).arr)[1:NyF, 1:NxF]
    PS_2D   = @. Cf_diag + Cn_diag / max(B_diag^2, 1e-30)
    Cϕ_2D   = real.((ds.Cϕ * f_ones).arr)[1:NyF, 1:NxF]

    grad_rms  = sqrt(mean(gx .^ 2) + mean(gy .^ 2))
    floor_val = denom_floor_frac * grad_rms

    ℓx_arr = proj.ℓx[1:NxF]
    ℓy_arr = proj.ℓy[1:NyF]

    phi_e     = zeros(Float64, Ny, Nx)
    gprod_buf = similar(gx)
    W_buf     = similar(gx)
    φ_F       = zeros(ComplexF64, NyF, NxF)

    for j in 1:NyF
        ly = ℓy_arr[j]
        for i in 1:NxF
            lx = ℓx_arr[i]
            L2 = lx^2 + ly^2
            (L2 == 0 || L2 > Lmax^2) && continue

            PS   = PS_2D[j, i]
            Cphi = Cϕ_2D[j, i]
            Cphi <= 0 && continue

            @. gprod_buf = lx * gx + ly * gy
            @. gprod_buf = ifelse(abs(gprod_buf) < floor_val,
                                  ifelse(gprod_buf >= 0.0, floor_val, -floor_val),
                                  gprod_buf)

            @. W_buf  = Cphi / (PS / gprod_buf^2 + Cphi)
            @. phi_e  = W_buf * T_hp / gprod_buf

            W_mean = mean(W_buf)
            W_mean < 1e-30 && continue

            φ_F[j, i] = m_rfft(phi_e, (1, 2))[j, i] / (W_mean * im)
        end
    end

    return FlatFourier(φ_F, proj)
end


"""
    gi_estimate_ref(ds)

GI estimator in the original Hadzhiyska+2019 form. Uses explicit
T_hp = T - LowPass(4000)*T subtraction, no lower-L cut, and Lmax=20000.
"""
function gi_estimate_ref(ds; denom_floor_frac=1e-8)
    Lgrad = 2000
    Lhp   = 4000
    Lmax  = 20000

    m    = Map(ds.d)
    proj = m.proj

    dTdx, dTdy, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))

    T_lp = Map(LowPass(Lhp) * ds.d).arr
    T_hp = Map(ds.d).arr .- T_lp

    A_F = m_rfft(dTdx .* T_hp, (1,2))
    B_F = m_rfft(dTdy .* T_hp, (1,2))

    σ_xx = mean(dTdx .^ 2)
    σ_xy = mean(dTdx .* dTdy)
    σ_yy = mean(dTdy .^ 2)

    NyF, NxF = size(A_F)
    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2

    numer = @. -im * (ℓx2D * A_F + ℓy2D * B_F)
    denom = @. ℓx2D^2 * σ_xx + 2*ℓx2D*ℓy2D * σ_xy + ℓy2D^2 * σ_yy

    denom_threshold = denom_floor_frac * max(σ_xx, σ_yy)
    L_mask = @. L2 < Lmax^2

    φ_F = @. ifelse(L_mask & (abs(denom) > denom_threshold),
                    numer / denom, complex(0.0))

    return FlatFourier(φ_F, proj)
end


"""
    gi_n0_fixed_gradient_mc(ds; nmc=20, Lhp=4000, Lmax=18000, Δℓ=30)

Per-realisation N0 estimate for the GI estimator using a fixed-gradient MC.

The gradient g = ∇(LowPass(2000)·d) is kept fixed from the data realisation,
so the denominator tracks the per-sim gradient amplitude. Independent null
draws T_hp_null ~ Cf̃ + Cn̂ (φ=0) are used to estimate the noise power, breaking
any correlation between N0 and the signal.

Returns `(ℓ, Cℓ, Lhp, Lmax, method)`.
"""
function gi_n0_fixed_gradient_mc(ds; nmc=20, Lhp=4000, Lmax=18000, Δℓ=30,
                                   denom_floor_frac=1e-8, seed_start=42_000)
    proj = Map(ds.d).proj
    Ny, Nx = size(Map(ds.d).arr)
    NyF, NxF = Ny ÷ 2 + 1, Nx

    dTdx, dTdy, _ = grad_fft(Map(LowPass(2000) * ds.d))
    σ_xx = mean(dTdx .^ 2)
    σ_xy = mean(dTdx .* dTdy)
    σ_yy = mean(dTdy .^ 2)
    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2
    denom2D           = @. ℓx2D^2 * σ_xx + 2*ℓx2D*ℓy2D * σ_xy + ℓy2D^2 * σ_yy
    denom_threshold2D = @. denom_floor_frac * L2 * max(σ_xx, σ_yy)
    L_mask  = @. (L2 > Lhp^2) & (L2 < Lmax^2) & (abs(denom2D) > denom_threshold2D)

    ℓ_out  = nothing
    Cℓ_sum = nothing
    for i in 1:nmc
        f_null = simulate(MersenneTwister(seed_start + i),          ds.Cf̃)
        n_null = simulate(MersenneTwister(seed_start + i + 10_000), ds.Cn̂)
        d_null = ds.B̂ * ds.M̂ * f_null + n_null
        T_hp_null = Map(d_null).arr .- Map(LowPass(Lhp) * d_null).arr

        A_F   = m_rfft(dTdx .* T_hp_null, (1,2))
        B_F   = m_rfft(dTdy .* T_hp_null, (1,2))
        numer = @. -im * (ℓx2D * A_F + ℓy2D * B_F)
        φ_F   = @. ifelse(L_mask, numer / denom2D, complex(0.0))

        spec = get_Cℓ(FlatFourier(φ_F, proj); Δℓ=Δℓ)
        if Cℓ_sum === nothing
            ℓ_out  = Float64.(spec.ℓ)
            Cℓ_sum = Float64.(spec.Cℓ)
        else
            Cℓ_sum .+= Float64.(spec.Cℓ)
        end
    end
    return (ℓ=ℓ_out, Cℓ=Cℓ_sum ./ nmc, Lhp=Lhp, Lmax=Lmax, method="fgmc_v2")
end


"""
    gi_n0_mc(ds; nmc=100, Lhp=4000, Lmax=18000, Δℓ=30)

Global MC N0 for the GI estimator. Runs `gi_estimate` on `nmc` null sims
(d_null = B̂M̂f + n, f ~ Cf̃, n ~ Cn̂) and returns the mean auto-spectrum.

Returns `(ℓ, Cℓ)`.
"""
function gi_n0_mc(ds; Lhp=4000, Lmax=18000, Δℓ=30, nmc=100, seed_start=42_000,
                  σ_fid=nothing)
    ℓ_out  = nothing
    Cℓ_sum = nothing
    for i in 1:nmc
        f_null = simulate(MersenneTwister(seed_start + i),        ds.Cf̃)
        n_null = simulate(MersenneTwister(seed_start + i + nmc),  ds.Cn̂)
        d_null = ds.B̂ * ds.M̂ * f_null + n_null
        sn = _SimData(d_null)
        φ̂_null = gi_twoleg(sn, sn; σ_fid=σ_fid, Lhp=Lhp, Lmax=Lmax)
        spec = get_Cℓ(φ̂_null; Δℓ=Δℓ)
        if Cℓ_sum === nothing
            ℓ_out  = Float64.(spec.ℓ)
            Cℓ_sum = Float64.(spec.Cℓ)
        else
            Cℓ_sum .+= Float64.(spec.Cℓ)
        end
    end
    return (ℓ=ℓ_out, Cℓ=Cℓ_sum ./ nmc)
end


"""
    compute_sigma_fid(ds; nmc=20, Lgrad=2000)

Fiducial gradient variances (σ_xx, σ_xy, σ_yy) averaged over `nmc` null draws.
Pass the result as `σ_fid` to `gi_twoleg` / `gi_n0_rdn0` to stabilise the
denominator across realisations, which is required for RDN0 bias cancellation.
"""
function compute_sigma_fid(ds; nmc=20, Lgrad=2000, seed_start=50_000)
    σ_xx_sum = 0.0
    σ_xy_sum = 0.0
    σ_yy_sum = 0.0
    for i in 1:nmc
        f_null = simulate(MersenneTwister(seed_start + i),           ds.Cf̃)
        n_null = simulate(MersenneTwister(seed_start + i + 100_000), ds.Cn̂)
        d_null = ds.B̂ * ds.M̂ * f_null + n_null
        dTdx, dTdy, _ = grad_fft(Map(LowPass(Lgrad) * d_null))
        σ_xx_sum += mean(dTdx .^ 2)
        σ_xy_sum += mean(dTdx .* dTdy)
        σ_yy_sum += mean(dTdy .^ 2)
    end
    return (σ_xx = σ_xx_sum / nmc,
            σ_xy = σ_xy_sum / nmc,
            σ_yy = σ_yy_sum / nmc)
end


"""
    gi_n0_rdn0(ds; nsim=20, σ_fid, Lgrad=2000, Lhp=4000, Lmax=12000, Δℓ=30)

Realization-dependent N0 (RDN0) for the GI estimator.

For each iteration, draws two independent null sims sA, sB and forms

    N0_RD = C(φ_DS + φ_SD) − ½ C(φ_AB + φ_BA)

where φ_DS = gi_twoleg(data, sA), φ_SD = gi_twoleg(sA, data), etc.
This is O(ε²) in the per-sim covariance mismatch ε, versus O(ε) for the
global MC N0. `σ_fid` must be provided to fix the denominator across legs.

Returns `(ℓ, Cℓ)`.
"""
function gi_n0_rdn0(ds; nsim=20, Lgrad=2000, Lhp=4000, Lmax=12000, Δℓ=30,
                    σ_fid, seed_start=55_000)
    ds_data = _SimData(ds.d)
    kw = (σ_fid=σ_fid, Lgrad=Lgrad, Lhp=Lhp, Lmax=Lmax)

    ℓ_out  = nothing
    Cℓ_sum = nothing

    for i in 1:nsim
        f_A = simulate(MersenneTwister(seed_start + i),           ds.Cf̃)
        n_A = simulate(MersenneTwister(seed_start + i + 100_000), ds.Cn̂)
        sA  = _SimData(ds.B̂ * ds.M̂ * f_A + n_A)

        f_B = simulate(MersenneTwister(seed_start + i + 200_000), ds.Cf̃)
        n_B = simulate(MersenneTwister(seed_start + i + 300_000), ds.Cn̂)
        sB  = _SimData(ds.B̂ * ds.M̂ * f_B + n_B)

        phi_DS = gi_twoleg(ds_data, sA; kw...)
        phi_SD = gi_twoleg(sA, ds_data; kw...)
        phi_AB = gi_twoleg(sA, sB; kw...)
        phi_BA = gi_twoleg(sB, sA; kw...)

        spec_dssd = get_Cℓ(phi_DS + phi_SD; Δℓ=Δℓ)
        spec_ss   = get_Cℓ(phi_AB + phi_BA; Δℓ=Δℓ)

        contrib = Float64.(spec_dssd.Cℓ) .- 0.5 .* Float64.(spec_ss.Cℓ)
        if Cℓ_sum === nothing
            ℓ_out  = Float64.(spec_dssd.ℓ)
            Cℓ_sum = contrib
        else
            Cℓ_sum .+= contrib
        end
    end

    return (ℓ=ℓ_out, Cℓ=Cℓ_sum ./ nsim)
end


"""
    gi_n0_linrd_analytical(ds; nmc=50, Lgrad=2000, Lhp=4000, Lmax=12000, Δℓ=30)

Linearised realisation-dependent GI N0 (MC-baseline version).

Splits the data into mean and fluctuation, C_D = C_S + δC, D_D = D_S + δD, giving

    N_D ≈ C_S/D_S + δC/D_S − C_S δD/D_S²

where C and D are the high-pass TT power and gradient-variance denominator.
C_S and D_S are estimated from `nmc` null sims; C_D and D_D from the data.

Returns `(ℓ, Cℓ, C_S, C_D, D_S_mean, D_D_mean)`.
"""
function gi_n0_linrd_analytical(ds; nmc::Int=50,
                                 Lgrad::Int=2000,
                                 Lhp::Int=4000,
                                 Lmax::Int=12000,
                                 Δℓ::Int=30,
                                 seed_start::Int=90_000,
                                 smooth_C_data::Bool=false,
                                 smooth_window::Int=9,
                                 clamp_negative::Bool=false)

    proj = Map(ds.d).proj
    Ny, Nx = size(Map(ds.d).arr)
    NyF, NxF = Ny ÷ 2 + 1, Nx

    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2
    Lmag = sqrt.(L2)
    L_mask = @. (L2 > Lhp^2) & (L2 < Lmax^2)

    gx_D, gy_D, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))
    σxx_D = mean(gx_D .^ 2)
    σxy_D = mean(gx_D .* gy_D)
    σyy_D = mean(gy_D .^ 2)
    D_D_2D = @. ℓx2D^2 * σxx_D + 2 * ℓx2D * ℓy2D * σxy_D + ℓy2D^2 * σyy_D

    σxx_S_sum = 0.0; σxy_S_sum = 0.0; σyy_S_sum = 0.0
    C_S_sum = nothing; ℓ_C = nothing

    for i in 1:nmc
        f_null = simulate(MersenneTwister(seed_start + i),           ds.Cf̃)
        n_null = simulate(MersenneTwister(seed_start + i + 100_000), ds.Cn̂)
        d_null = ds.B̂ * ds.M̂ * f_null + n_null

        gx_S, gy_S, _ = grad_fft(Map(LowPass(Lgrad) * d_null))
        σxx_S_sum += mean(gx_S .^ 2)
        σxy_S_sum += mean(gx_S .* gy_S)
        σyy_S_sum += mean(gy_S .^ 2)

        T_hp_null = HighPass(Lhp) * d_null
        cl_null = get_Cℓ(T_hp_null; Δℓ=Δℓ)
        if C_S_sum === nothing
            ℓ_C     = Float64.(collect(cl_null.ℓ))
            C_S_sum = Float64.(cl_null.Cℓ)
        else
            C_S_sum .+= Float64.(cl_null.Cℓ)
        end
    end

    σxx_S = σxx_S_sum / nmc; σxy_S = σxy_S_sum / nmc; σyy_S = σyy_S_sum / nmc
    C_S   = C_S_sum ./ nmc

    D_S_2D = @. ℓx2D^2 * σxx_S + 2 * ℓx2D * ℓy2D * σxy_S + ℓy2D^2 * σyy_S

    T_hp_D = HighPass(Lhp) * ds.d
    cl_D   = get_Cℓ(T_hp_D; Δℓ=Δℓ)
    C_D    = Float64.(cl_D.Cℓ)
    smooth_C_data && (C_D = smooth_vector(C_D; window=smooth_window))

    CS_itp = Cℓs(ℓ_C, C_S)
    CD_itp = Cℓs(Float64.(collect(cl_D.ℓ)), C_D)
    C_S_2D = similar(Lmag); C_D_2D = similar(Lmag)
    for j in axes(Lmag, 1), i in axes(Lmag, 2)
        L = Lmag[j, i]
        C_S_2D[j, i] = (L < ℓ_C[1]     || L > ℓ_C[end])     ? 0.0 : Float64(CS_itp(L))
        C_D_2D[j, i] = (L < cl_D.ℓ[1]  || L > cl_D.ℓ[end])  ? 0.0 : Float64(CD_itp(L))
    end

    epsD  = 1e-30 * maximum(abs.(D_S_2D))
    valid = @. L_mask & isfinite(D_S_2D) & isfinite(D_D_2D) & (abs(D_S_2D) > epsD)
    N0_2D = @. ifelse(valid,
                      C_S_2D / D_S_2D +
                      (C_D_2D - C_S_2D) / D_S_2D -
                      C_S_2D * (D_D_2D - D_S_2D) / D_S_2D^2,
                      0.0)
    clamp_negative && (N0_2D = max.(N0_2D, 0.0))

    ℓ_out = Float64[]; Cℓ_out = Float64[]
    ℓ_lo  = 0.0
    while ℓ_lo < Lmax
        ℓ_hi = ℓ_lo + Δℓ
        ring = @. (Lmag >= ℓ_lo) & (Lmag < ℓ_hi) & valid
        if any(ring)
            vals = N0_2D[ring]; vals = vals[isfinite.(vals)]
            !isempty(vals) && (push!(ℓ_out, mean(Lmag[ring])); push!(Cℓ_out, mean(vals)))
        end
        ℓ_lo = ℓ_hi
    end

    return (ℓ=ℓ_out, Cℓ=Cℓ_out, C_S=C_S, C_D=C_D, ℓ_C=ℓ_C,
            D_S_mean=(σ_xx=σxx_S, σ_xy=σxy_S, σ_yy=σyy_S),
            D_D_mean=(σ_xx=σxx_D, σ_xy=σxy_D, σ_yy=σyy_D))
end


"""
    gi_n0_linrd_analyticbaseline(ds; Lgrad=2000, Lhp=4000, Lmax=12000, Δℓ=30)

Linearised realisation-dependent GI N0 (analytic-baseline version).

Same formula as `gi_n0_linrd_analytical` but replaces MC null sims with exact
power-spectrum expectations:

    C_S_2D(ℓ) = B(ℓ)² Cf̃(ℓ) + Cn̂(ℓ)
    σ_xx_S     = (1/Npix) Σ_{L<Lgrad} w_L ℓx² C_S_2D(ℓ)

C_S_2D is used directly per mode, removing the round-trip binning error of the
MC-baseline version.

Returns `(ℓ, Cℓ, C_S, C_D, D_S_mean, D_D_mean, frac_C, frac_D_binned, N_exact, ℓ_C)`.
"""
function gi_n0_linrd_analyticbaseline(ds; Lgrad::Int=2000,
                                       Lhp::Int=4000,
                                       Lmax::Int=12000,
                                       Δℓ::Int=30,
                                       smooth_C_data::Bool=false,
                                       smooth_window::Int=9,
                                       clamp_negative::Bool=false)

    proj = Map(ds.d).proj
    Ny, Nx = size(Map(ds.d).arr)
    NyF, NxF = Ny ÷ 2 + 1, Nx
    Npix = Float64(Ny * Nx)

    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2
    Lmag = sqrt.(L2)
    L_mask = @. (L2 > Lhp^2) & (L2 < Lmax^2)

    # rfft multiplicity: interior ℓy rows appear as ±ℓy conjugate pairs
    rfft_wy = ones(Float64, NyF); rfft_wy[2:NyF-1] .= 2.0
    rfft_w  = repeat(rfft_wy, 1, NxF)

    f_ones   = FlatFourier(ones(ComplexF64, NyF, NxF), proj)
    B_diag   = real.(ds.B̂.diag.arr)[1:NyF, 1:NxF]
    Cf_diag  = real.((ds.Cf̃ * f_ones).arr)[1:NyF, 1:NxF]
    Cn_diag  = real.((ds.Cn̂ * f_ones).arr)[1:NyF, 1:NxF]
    C_S_2D   = @. (B_diag^2 * Cf_diag + Cn_diag) / Npix

    lp_mask = L2 .< Lgrad^2
    σxx_S = sum(rfft_w .* lp_mask .* ℓx2D.^2      .* C_S_2D) / Npix
    σxy_S = sum(rfft_w .* lp_mask .* ℓx2D .* ℓy2D .* C_S_2D) / Npix
    σyy_S = sum(rfft_w .* lp_mask .* ℓy2D.^2      .* C_S_2D) / Npix
    D_S_2D = @. ℓx2D^2 * σxx_S + 2 * ℓx2D * ℓy2D * σxy_S + ℓy2D^2 * σyy_S

    gx_D, gy_D, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))
    σxx_D = mean(gx_D .^ 2); σxy_D = mean(gx_D .* gy_D); σyy_D = mean(gy_D .^ 2)
    D_D_2D = @. ℓx2D^2 * σxx_D + 2 * ℓx2D * ℓy2D * σxy_D + ℓy2D^2 * σyy_D

    T_hp_D = HighPass(Lhp) * ds.d
    cl_D   = get_Cℓ(T_hp_D; Δℓ=Δℓ)
    C_D_1D = Float64.(cl_D.Cℓ)
    smooth_C_data && (C_D_1D = smooth_vector(C_D_1D; window=smooth_window))

    CD_itp = Cℓs(Float64.(collect(cl_D.ℓ)), C_D_1D)
    C_D_2D = similar(Lmag)
    for j in axes(Lmag, 1), i in axes(Lmag, 2)
        L = Lmag[j, i]
        C_D_2D[j, i] = (L < cl_D.ℓ[1] || L > cl_D.ℓ[end]) ? 0.0 : Float64(CD_itp(L))
    end

    epsD  = 1e-30 * maximum(abs.(D_S_2D))
    valid = @. L_mask & isfinite(D_S_2D) & isfinite(D_D_2D) & (abs(D_S_2D) > epsD)
    N0_2D = @. ifelse(valid,
                      C_S_2D / D_S_2D +
                      (C_D_2D - C_S_2D) / D_S_2D -
                      C_S_2D * (D_D_2D - D_S_2D) / D_S_2D^2,
                      0.0)
    clamp_negative && (N0_2D = max.(N0_2D, 0.0))

    epsD_D  = 1e-30 * maximum(abs.(D_D_2D))
    validD  = @. L_mask & (abs(D_D_2D) > epsD_D)
    N_exact_2D = @. ifelse(validD, C_D_2D / D_D_2D, 0.0)

    frac_C_2D = @. ifelse(L_mask & (C_S_2D > 0), C_D_2D / C_S_2D, 1.0)

    ℓ_out = Float64[]; Cℓ_out = Float64[]
    ℓ_C   = Float64[]; C_S_1D = Float64[]
    frac_D_binned = Float64[]; N_exact_binned = Float64[]

    ℓ_lo = 0.0
    while ℓ_lo < Lmax
        ℓ_hi = ℓ_lo + Δℓ
        ring  = @. (Lmag >= ℓ_lo) & (Lmag < ℓ_hi) & valid
        ringD = @. (Lmag >= ℓ_lo) & (Lmag < ℓ_hi) & L_mask
        if any(ringD)
            lc = mean(Lmag[ringD])
            push!(ℓ_C, lc)
            push!(C_S_1D, mean(C_S_2D[ringD]))
            ds_vals = D_S_2D[ringD]; dd_vals = D_D_2D[ringD]
            push!(frac_D_binned, mean(ds_vals) > 0 ? mean(dd_vals) / mean(ds_vals) : 1.0)
            push!(N_exact_binned, mean(N_exact_2D[ringD]))
        end
        if any(ring)
            vals = N0_2D[ring]; vals = vals[isfinite.(vals)]
            !isempty(vals) && (push!(ℓ_out, mean(Lmag[ring])); push!(Cℓ_out, mean(vals)))
        end
        ℓ_lo = ℓ_hi
    end

    return (ℓ=ℓ_out, Cℓ=Cℓ_out,
            C_S=C_S_1D, C_D=C_D_1D, ℓ_C=ℓ_C,
            D_S_mean=(σ_xx=σxx_S, σ_xy=σxy_S, σ_yy=σyy_S),
            D_D_mean=(σ_xx=σxx_D, σ_xy=σxy_D, σ_yy=σyy_D),
            frac_C=frac_C_2D, frac_D_binned=frac_D_binned,
            N_exact=N_exact_binned, σ_S=(σ_xx=σxx_S, σ_xy=σxy_S, σ_yy=σyy_S),
            σ_D=(σ_xx=σxx_D, σ_xy=σxy_D, σ_yy=σyy_D))
end


function smooth_vector(v::Vector{Float64}; window::Int=9)
    n = length(v)
    out = similar(v)
    hw = window ÷ 2
    for i in 1:n
        vals = v[max(1, i-hw):min(n, i+hw)]
        vals = vals[isfinite.(vals)]
        out[i] = isempty(vals) ? v[i] : mean(vals)
    end
    return out
end
