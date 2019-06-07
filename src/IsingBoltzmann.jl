### NOTE: The Ising model uses spins ∈ {+-1}; the Boltzmann machine uses variables ∈ {0,1}.

module IsingBoltzmann
## stdlib ######################################################################
using Random
################################################################################
## Individual ##################################################################
using Reexport: @reexport
using Requires: @require

export ising, rbm, cpu_rbm, train

function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuarrays.jl")
end

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
    kernel_kwargs::Dict{Symbol, Any}

    nepochs::Int
    nsamples::Int
    batchsize::Int
    rng::AbstractRNG

    cuarrays::Bool=false
    curng::Any=nothing
end
propertynames(config::AppConfig, ::Bool) = (:ndims, :nspins, fieldnames(config)...)
Base.getproperty(config::AppConfig, s::Symbol) =
    if s === :ndims
        length(getfield(config, :spinsize))
    elseif s === :nspins
        prod(getfield(config, :spinsize))
    else
        getfield(config, s)
    end

ising(config) =
    IsingModel(config.boundarycond, config.spinsize;
        coupling=config.coupling, invtemp=config.invtemp
    )

rbm(config) =
    if config.cuarrays
        cuarrays_rbm(config)
    else
        cpu_rbm(config)
    end

function cpu_rbm(config)
    init(dims) =
        sqrt(inv(config.nspins+config.nhiddens)).*2.0.*(rand(config.rng, dims...) .- 0.5)

    RestrictedBoltzmann(config.nspins, config.nhiddens; init=init)
end

_nothing(_...) = nothing

train(config) = _train(_nothing, ising(config), config)
train(cb, config) = _train(cb, ising(config), config)
train(cb, ising, config) = _train(cb, ising, config)
function _train(cb, ising, config)
    rbmachine = rbm(config)
    local kern
    try
        kern = config.kldiv_grad_kernel(rbmachine; config.kernel_kwargs...)
    catch ex
        ex isa MethodError && ex.f == config.kldiv_grad_kernel || rethrow()
        kwparams_str = join(string.(keys(config.kernel_kwargs)), ", ")
        error(
            "Don't know how to build kernel ", config.kldiv_grad_kernel, " from given ",
            "configuration.\n\n",
            "Provide a method ", config.kldiv_grad_kernel,"(::", typeof(rbmachine), ";",
            kwparams_str, ") or construct RBM and kernel explicitly and pass to train!()."
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
