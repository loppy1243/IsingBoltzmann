### NOTE: The Ising model uses spins ∈ {+-1}; the Boltzmann machine uses variables ∈ {0,1}.
### These have to be transformed between properly!

module IsingBoltzmann
## stdlib ######################################################################
using Random
################################################################################
## Individual ##################################################################
using Reexport: @reexport

export ising, rbm, train

include("utils.jl")
include("MetropolisHastings.jl"); @reexport using .MetropolisHastings
include("PeriodicArrays.jl")
include("Spins.jl");              @reexport using .Spins
include("Ising.jl");              @reexport using .Ising
include("RBM.jl");                @reexport using .RBM

Base.@kwdef mutable struct AppConfig
    spinsize::Dims
    coupling::Float64
    invtemp::Float64
    boundarycond::Ising.BoundaryCondition

    nhiddens::Int
    learning_rate::Float64
    cd_num::Int
    kldiv_grad_kernel::Type{<:KLDivGradKernel}

    nepochs::Int
    sample_epochs::Vector{Int}
    nsamples::Int
    batchsize::Int
    rng::AbstractRNG
end
propertynames(config::AppConfig, ::Bool) = (fieldnames(config)..., :ndims, :nspins)
Base.getproperty(config::AppConfig, s::Symbol) =
    if s === :ndims
        length(getfield(config, :spinsize))
    elseif s === :nspins
        prod(getfield(config, :spinsize))
    elseif s === :n_sample_epochs
        length(getfield(config, :sample_epochs))
    else
        getfield(config, s)
    end

ising(config) =
    IsingModel(config.boundarycond, config.spinsize;
        coupling=config.coupling, invtemp=config.invtemp
    )

function rbm(config)
    init(dims) =
        sqrt(inv(config.nspins+config.nhiddens)).*2.0.*(rand(config.rng, dims...) .- 0.5)

    RestrictedBoltzmann(config.nspins, config.nhiddens;
        init=init, learning_rate=config.learning_rate, cd_num=config.cd_num
    )
end

_nothing(_...) = nothing

train(config) = _train(_nothing, ising(config), config)
train(cb, config) = _train(cb, ising(config), config)
train(cb, ising, config) = _train(cb, ising, config)
function _train(cb, ising, config)
    rbmachine = rbm(config)
    kern = if hasmethod(config.kldiv_grad_kernel, Tuple{RestrictedBoltzmann})
        config.kldiv_grad_kernel(rbmachine)
    else
        error(
            "Don't know how to build kernel ", config.kldiv_grad_kernel, " from given ",
            "configuration.\n\n",
            "Provide a method ", config.kldiv_grad_kernel, "(::RestrictedBoltzmann) or ",
            "construct RBM and kernel explicitly and pass to train!()."
        )
    end

    metro = MetropolisIsingSampler(ising; init=dims -> spinrand(config.rng, dims))
    ising_samples = rand(config.rng, metro, config.nsamples)

    _train!(
        cb, config.rng, rbmachine, kern, ising, ising_samples, config.batchsize,
        config.nepochs
    )
end

train!(args...) = _train!(_nothing, Random.GLOBAL_RNG, args...)
train!(cb::Function, args...) = _train!(cb, Random.GLOBAL_RNG, args...)
train!(rng::AbstractRNG, args...) = _train!(_nothing, rng, args...)
train!(kern::KLDivGradKernel, args...) = _train!(_nothing, rng, args...)
train!(cb::Function, rng::AbstractRNG, args...) = _train!(rng, cb, args...)
function _train!(cb, rng, rbm, kern, ising, ising_samples, batchsize, nepochs)
    batches = batch(map(vec, ising_samples), batchsize)
    for epoch = 1:nepochs
        cb(epoch-1, nepochs, rbm, ising, ising_samples)
        RBM.train!(rng, rbm, kern, batches)
    end
    cb(nepochs, nepochs, rbm, ising, ising_samples)

    rbm, kern
end

end # module IsingBoltzmann
