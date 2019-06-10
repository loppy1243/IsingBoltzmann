module IsingBoltzmannRun2D
## Internal ####################################################################
using IsingBoltzmann
## stdlib ######################################################################
import Dates
## External ####################################################################
import Flux.Optimise
using Plots
################################################################################
## Individual: Internal ########################################################
using IsingBoltzmann: bitstrings, batch
## stdlib ######################################################################
using Random: MersenneTwister, shuffle!
using Printf: @sprintf
using Statistics: mean

DEFAULT_CONFIG() = IsingBoltzmann.AppConfig(
    spinsize=(8,8),
    coupling=1.0,
    invtemp=0.4,
    boundarycond=Ising.FixedBoundary(),
    
    nhiddens=32,
    kldiv_grad_kernel=KLDivGradKernels.ExactKernel,
    kernel_kwargs=Dict(:cd_num=>5),
    optimizer=Optimise.Descent(0.01),

    rng=MersenneTwister(1022150136457226227)
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

    kldiv_plotfile::String="kldiv_2D.pdf"
    pdf_plotfile::String="pdf_1D.pdf"

    debug::Bool=false
    plot::Bool=true
end

function __init__()
    Plots.gr()
    nothing
end

tuplecomp((a1, b1), (a2, b2)) = a1 < a2 || a1 == a2 && b1 <= b2

main(; kwargs...) = main(LocalConfig(; kwargs...), DEFAULT_CONFIG())
function main(LCONFIG, CONFIG)
    if LCONFIG.debug
        print("LCONFIG = ")
        show(stdout, "text/plain", LCONFIG)
        print("\nCONFIG = ")
        show(stdout, "text/plain", CONFIG)
        println()
    end

    isingmodel = ising(CONFIG)
    ising_sampler = MetropolisIsingSampler(isingmodel; init=dims->spinrand(CONFIG.rng, dims))
    ising_samples = rand(CONFIG.rng, ising_sampler, LCONFIG.nsamples)

    kldivs_exact = Dict{NTuple{2, Int}, Float64}()
    kldivs_approx = Dict{NTuple{2, Int}, Float64}()

    cb = callback(
        kldivs_exact, kldivs_approx, LCONFIG, CONFIG
    )

    LCONFIG.debug && println("Creating and training RBM:")
    ising_minibatches = batch(vec.(ising_samples), LCONFIG.minibatchsize)
    train(cb, LCONFIG.nepochs, ()->shuffle!(CONFIG.rng, ising_minibatches), CONFIG)

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

    nothing
end

callback(kldivs_exact, kldivs_approx, LCONFIG, CONFIG) =
function(epoch, minibatchnum, rbm, minibatch)
    epochfmt(epoch) = lpad(epoch, ndigits(LCONFIG.nepochs))

    idx = (epoch, minibatchnum)

    if all(in.(idx, Tuple(LCONFIG.kldiv_samples)))
        println("Epoch ", epochfmt(epoch))
    end

    false
end

end # module IsingBoltzmannRun2D
