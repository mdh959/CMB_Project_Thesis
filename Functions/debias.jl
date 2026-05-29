function debias_phi_with_WL(ϕ, WL_Cℓs; minW::Real=1e-8)
    ϕF   = Fourier(ϕ)
    ℓmag = Float64.(ϕF.metadata.ℓmag)

    ℓmin = minimum(WL_Cℓs.ℓ)
    ℓmax = maximum(WL_Cℓs.ℓ)
    ℓcl  = clamp.(ℓmag, ℓmin, ℓmax)

    W = Float64.(WL_Cℓs.(ℓcl))

    out = similar(ϕF.arr)
    @inbounds for i in eachindex(out)
        w = W[i]
        if !isfinite(w) || w <= minW
            out[i] = 0.0
        else
            out[i] = ϕF.arr[i] / w
        end
    end

    return typeof(ϕF)(out, ϕF.metadata)
end