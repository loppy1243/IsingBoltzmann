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
    kldiv_grad_kernel::Type{<:KLDivGradKernel}
    kernel_kwargs::Dict{Symbol, Any}
    # Flux doesn't have an abstract type for optimizers...
    optimizer::Any

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

_nothing(_...) = false

train(minibatches, config) = train(_nothing, 1, minibatches, config)
train(nepochs, minibatches, config) = train(_nothing, nepochs, minibatches, config)
function train(cb, nepochs, minibatches, config)
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

    rng = config.cuarrays ? config.curng : config.rng
    train!(cb, rng, nepochs, rbmachine, kern, config.optimizer, minibatches)
end

let _Args = Vararg{Any, 5}
    global train!
    train!(args::_Args) = train!(_nothing, GLOBAL_RNG, args...)
    train!(rng::AbstractRNG, args::_Args) = train!(_nothing, rng, args...)
end
train!(cb::Function, rng::AbstractRNG, nepochs, rbm, kern, opt, minibatches) =
    train!(cb, rng, nepochs, rbm, kern, opt, nepochs, _->minibatches)
function train!(cb::Function, rng::AbstractRNG, nepochs, rbm, kern, opt, nextbatch::Function)
    done = false
    for epoch = 1:nepochs
        minibatches = nextbatch()
        cb(epoch, 0, rbm, first(minibatches)) && break
        for (n, mb) in enumerate(minibatches)
            RBM.update!(rng, rbm, kern, opt, mb)
            cb(epoch, n, rbm, mb) && (done = true; break)
        end
        done && break
    end

    rbm, kern
end

end # module IsingBoltzmann
