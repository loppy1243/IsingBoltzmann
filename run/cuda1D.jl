module IsingBoltzmannCudaRun1D
## Internal ####################################################################
using IsingBoltzmann
## stdlib ######################################################################
import Dates
## External ####################################################################
import Flux.Optimise, CuArrays.CURAND, CUDAnative
using Plots
################################################################################
## Individual: Internal ########################################################
using IsingBoltzmann: bitstrings, batch
## stdlib ######################################################################
using Random: MersenneTwister, shuffle!
using Printf: @sprintf
using Statistics: mean

global CURNG::CURAND.RNG
DEFAULT_CONFIG() = IsingBoltzmann.AppConfig(
    spinsize=(6,),
    coupling=1.0,

    invtemp=0.4,
    boundarycond=Ising.FixedBoundary(),
    
    nhiddens=6,
    kldiv_grad_kernel=KLDivGradKernels.ExactKernel,
    kernel_kwargs=Dict(:cd_num=>5),
    optimizer=Optimise.Descent(0.01),

    rng=MersenneTwister(1022150136457226227),

    cuarrays=true,
    curng=CURNG
)

Base.@kwdef mutable struct LocalConfig
    nsamples::Int=10^5
    minibatchsize::Int=50
    nminibatches::Int=div(nsamples, minibatchsize)
    nepochs::Int=1000

    kldiv_samples=(epochs=1:nepochs,    minibatches=nminibatches)
    pdf_samples  =(epochs=(10, 30, 50), minibatches=nminibatches)

    n_Δkldiv_avg_samples::Int=20
    Δkldiv_avg_atol::Float64=1e-2

    kldiv_plotfile::String="kldiv_1D.pdf"
    pdf_plotfile::String="pdf_1D.pdf"

    debug::Bool=false
    plot::Bool=true
end

## Based on https://github.com/JuliaGPU/CuArrays.jl/pull/344
function makecurng(seed, offset)
    rng = CURAND.generator()
    CURAND.set_pseudo_random_generator_seed(rng, seed)
    CURAND.set_generator_offset(rng, offset)
    CURAND.generate_seeds(rng)

    rng
end

function __init__()
    Plots.gr()

    CUDAnative.initialize()
    global CURNG = makecurng(228106312280517615, 647)

    nothing
end

tuplecomp((a1, b1), (a2, b2)) = a1 < a2 || a1 == a2 && b1 <= b2

main(; kwargs...) = main(LocalConfig(; kwargs...), DEFAULT_CONFIG())
function main(LCONFIG, CONFIG)
    isingmodel = ising(CONFIG)
    ising_pf = Ising.partitionfunc(isingmodel)
    ising_sampler = MetropolisIsingSampler(isingmodel; init=dims->spinrand(CONFIG.rng, dims))
    ising_samples = rand(CONFIG.rng, ising_sampler, LCONFIG.nsamples)

    prob_rbm = Dict{NTuple{2, Int}, Vector{Float64}}()
    kldivs_exact = Dict{NTuple{2, Int}, Float64}()
    kldivs_approx = Dict{NTuple{2, Int}, Float64}()

    cb = callback(
        isingmodel, ising_pf, prob_rbm, kldivs_exact, kldivs_approx, LCONFIG, CONFIG
    )

    LCONFIG.debug && println("Creating and training RBM:")
    train(cb, LCONFIG.nepochs, cu_minibatches_gen(ising_samples), LCONFIG, CONFIG)

    LCONFIG.debug && print("Creating kldiv plot...")

    kldivs_points = sort(collect(keys(kldivs_exact)); lt=tuplecomp)
    kldivs_exact_vals = [kldivs_exact[key] for key in kldivs_points]
    kldivs_approx_vals = [kldivs_approx[key] for key in kldivs_points]

    format((a, b)) = string(b, "\n", a)
    plot(format.(kldivs_points), [kldivs_exact_vals kldivs_approx_vals],
        label=["Exact" "Approx"],
        yscale=:log10,
        title="Ising 1D RBM KL Divergence", xlabel="Epoch", ylabel="KL Divergence"
    )
    annotate!([(15, 0, text(Dates.now(), "monospace", 12))])
    savefig(LCONFIG.kldiv_plotfile)
    LCONFIG.debug && println(" Done.")

    LCONFIG.plot || return nothing

    LCONFIG.debug && print("Creating PDF plot...")
    prob_exact = Ising.pdf.(Ref(isingmodel), spinstates(isingmodel); pfunc=ising_pf)
    first = true
    plts = []
    for key in sort(collect(keys(prob_rbm)); lt=tuplecomp)
        plt = plot(1:2^CONFIG.nspins, prob_exact, label="Exact",
            yscale = :log10, markershape=:auto,
            title="(Epochs, Samples)=$key", xlabel="\$\\sigma\$", ylabel="\$P(\\sigma)\$",
            legend=(first && (:top))
        )
        plot!(1:2^CONFIG.nspins, prob_rbm[key], label="RBM", markershape=:auto)

        push!(plts, plt)
        first = false
    end
    sz = Plots.default(:size)
    plot(plts...,
         layout=(length(prob_rbm), 1),
         size=(sz[1], length(prob_rbm)*sz[2])
    )
    annotate!([(0, 0, text(Dates.now(), "monospace", 12))])
    savefig(LCONFIG.pdf_plotfile)
    LCONFIG.debug && println(" Done.")

    nothing
end

function cu_minibatches_gen(ising_samples, LCONFIG, CONFIG)
    cu_ising_samples = map(σ -> CuVector{Bool}(undef, size(σ)), ising_samples)
    cu_minibatches = batch(cu_ising_samples, LCONFIG.minibatchsize)

    () -> Random.shuffle!(CONFIG.rng, cu_minibatches)
end

function callback(
        ising, ising_pf, prob_rbm, kldivs_exact, kldivs_approx, LCONFIG, CONFIG
)
    c_rbm = cpu_rbm(CONFIG)
    kld_exact_prev = nothing
    kld_approx_prev = nothing

    full = false
    Δidx = 1
    Δklds = Vector{Float64}(undef, LCONFIG.n_Δkldiv_avg_samples)
function(epoch, minibatchnum, g_rbm, minibatch)
    epochfmt(epoch) = lpad(epoch, ndigits(LCONFIG.nepochs))
    numfmt(num) = @sprintf("%.5f", num)
    deltafmt(num) = @sprintf("%+.5f", num)

    ret = false
    idx = (epoch, minibatchnum)

    LCONFIG.debug && print("    Copying RBM from GPU...")
    copyweights!(c_rbm, g_rbm)
    LCONFIG.debug && println(" Done.")

    if all(in.(idx, Tuple(LCONFIG.kldiv_samples)))
        rbm_pf = RBM.partitionfunc(c_rbm)

        LCONFIG.debug && print("    Computing kldiv_exact...")
        kld_exact = kldiv(c_rbm, Ising.pdf(ising; pfunc=ising_pf); pfunc=rbm_pf)
        LCONFIG.debug && println(" Done.")

        LCONFIG.debug && print("    Computing kldiv_approx...")
        kld_approx = kldiv(c_rbm, minibatch; pfunc=rbm_pf)
        LCONFIG.debug && println(" Done.")

        Δexact  = isnothing(kld_exact_prev)  ? 0.0 : kld_exact - kld_exact_prev
        Δapprox = isnothing(kld_approx_prev) ? 0.0 : kld_approx - kld_approx_prev
        println(
            epochfmt(epoch), ": ",
            "KL_approx=$(numfmt(kld_approx)), KL_exact=$(numfmt(kld_exact)), ",
            "ΔKL_approx=$(deltafmt(Δapprox)), ΔKL_exact=$(deltafmt(Δexact))"
        )

        kld_exact_prev = kld_exact; kld_approx_prev = kld_approx
        kldivs_exact[idx] = kld_exact; kldivs_approx[idx] = kld_approx

        Δklds[Δidx] = kld_exact

        Δidx == LCONFIG.n_Δkldiv_avg_samples && (full = true)
        Δidx = Δidx % LCONFIG.n_Δkldiv_avg_samples + 1
        full && (ret = ret || isapprox(mean(Δklds), 0.0; atol=LCONFIG.Δkldiv_avg_atol))
    end

    if all(in.(idx, Tuple(LCONFIG.pdf_samples)))
        rbm_pf = RBM.partitionfunc(c_rbm)

        LCONFIG.debug && print("    Calculating RBM PDF for epoch ", epoch, "...")
        prob_rbm[idx] = RBM.input_pdf.(Ref(c_rbm), bitstrings(c_rbm.inputsize); pfunc=rbm_pf)
        LCONFIG.debug && println(" Done.")
    end

    ret
end end

end # module IsingBoltzmannCudaRun1D
