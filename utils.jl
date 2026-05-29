module Utils

using CMBLensing, PythonPlot, NPZ, Statistics, LinearAlgebra, JLD2

export get_Cℓ_fft, bin_spectrum, bin_stat, bin_stat_err
export grad_fft
export gi_estimate, gi_full, gi_twoleg
export gi_n0_fixed_gradient_mc, gi_n0_mc,
       gi_n0_rdn0, gi_n0_linrd_analytical, gi_n0_linrd_analyticbaseline, compute_sigma_fid
export debias_phi_with_WL
export save_results
export plot_grad_T_lensed, compare_gradT_phi_errors,
       plot_phi_error_vs_gradT, plot_WL_comparison,
       plot_correlation_coefficient

include("Functions/spectra.jl")
include("Functions/gradient.jl")
include("Functions/gradient_inversion.jl")
include("Functions/debias.jl")
include("Functions/io.jl")
include("Functions/plotting.jl")

end
