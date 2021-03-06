module Ising
## Internal ####################################################################
using ..MetropolisHastings, ..Spins, ..PeriodicArrays
## stdlib ######################################################################
import Random
################################################################################
## Individual: Internal ########################################################
using ..IsingBoltzmann: bitstrings

export IsingModel, MetropolisIsingSampler, AbstractSpinGrid, boundarycond, worldtype,
       hamiltonian, partitionfunc, neighborhood, spinstates
# unexported API: BoundaryCondition, FixedBoundary, PeriodicBoundary, ndims, size, pdf

# D = Dimensions, B = BoundaryCondition
mutable struct IsingModel{D, B}
    size::Dims{D}
    coupling::Float64
    invtemp::Float64

    IsingModel{D, B}(sz::Dims{D}; coupling, invtemp) where {D, B} =
        new(sz, coupling, invtemp)
end
IsingModel{D, B}(sz::Int...; kwargs...) where {D, B} = IsingModel{D, B}(sz; kwargs...)
IsingModel(b, sz::Dims;   kwargs...) = IsingModel{length(sz), typeof(b)}(sz; kwargs...)
IsingModel(b, sz::Int...; kwargs...) = IsingModel(b, sz; kwargs...)

boundarycond(::Type{<:IsingModel{<:Any, B}}) where B = B
boundarycond(m::IsingModel) = boundarycond(typeof(m))

abstract type BoundaryCondition end
struct FixedBoundary <: BoundaryCondition end
struct PeriodicBoundary <: BoundaryCondition end

const AbstractSpinGrid{D} = AbstractArray{Spin, D}
struct MetropolisIsingSampler{D, I<:IsingModel{D}, W<:AbstractSpinGrid{D}} #=
       =# <: MetropolisHastingsSampler{W}
    model::I
    world::W

    # Flip *at most* this many spins
    stepsize::Int
    skip::Int

    function MetropolisIsingSampler{D, I, W}(
            model::I, world::W; stepsize=1, skip=0
    ) where {D, I<:IsingModel{D}, W<:AbstractSpinGrid{D}, R}
        @assert model.size == Base.size(world)

        new(model, world, stepsize, skip)
    end
end
const MIsingSampler = MetropolisIsingSampler
function MIsingSampler(
        model::IsingModel{D}, world::AbstractSpinGrid{D}; stepsize=1, skip=0
) where D
    MIsingSampler{D, typeof(model), typeof(world)}(
        model, world; stepsize=stepsize, skip=skip
    )
end
MetropolisHastings.StepType(::Type{<:MIsingSampler}) = MetropolisHastings.Reversible()

function MIsingSampler(model::IsingModel; init=falses, kws...)
    world = init(Ising.size(model))
    MIsingSampler{Ising.ndims(model), typeof(model), typeof(world)}(
        model, world; kws...
    )
end
MetropolisHastingsSampler(model::IsingModel) = MIsingSampler(model)

function Base.show(io::IO, mime::MIME"text/plain", m::MIsingSampler)
    println(io,
        "MetropolisIsingSampler(..., IsingModel($(m.model.coupling), $(m.model.invtemp)), ",
        "$(m.stepsize), $(m.skip))"
    )
    show(io, mime, m.world)
end

worldtype(::Type{<:MIsingSampler{<:Any, <:Any, W}}) where W = W
worldtype(m::MIsingSampler) = worldtype(typeof(m))

## Define on Base.ndims?
ndims(::Type{<:IsingModel{D}}) where D = D
ndims(::Type{<:MIsingSampler{D}}) where D = D
ndims(d) = ndims(typeof(d))
ndims(T::Type) = throw(MethodError(ndims, T))

## Define on Base.size?
size(m::IsingModel) = m.size
size(m::MIsingSampler) = size(m.model)

nspins(m) = prod(Ising.size(m))

spinstates(m) = spinstates(BitArray{Ising.ndims(m)}, m)
spinstates(::Type{BitArray{D}}, m::IsingModel{D}) where D =
    (reshape(bits, Ising.size(m)) for bits in bitstrings(nspins(m)))
spinstates(T::Type, m) =
    (convert(T, state) for state in spinstates(m))

hamiltonian(m::MIsingSampler) = hamiltonian(m.model, m.world)
function hamiltonian(m::IsingModel, x)
    ret = zero(m.coupling)
    for i in eachindex(x)
        for n in neighborhood(m, x, i)
            ret += 1 - 2*xor(x[i], n)
        end
    end

    -m.coupling*ret
end

hamildiff(m::MIsingSampler, ixs) = hamildiff(m.model, m.world, ixs)
## We get an _overall_ 2 since we have to consider the contribution of the site that flipped
## and its neighbors' contributions
function hamildiff(m::IsingModel, x, ixs)
    ret = zero(m.coupling)
    for i in ixs
        for n in neighborhood(m, x, i)
            ret += 2(xor(flipspin(x[i]), n) - xor(x[i], n))
        end
    end

    -2m.coupling*ret
end

## Can be generalized to a @generated function
neighborhood(m::IsingModel{1, FixedBoundary}, x::AbstractSpinGrid{1}, i) =
    if i == 1
        [x[i+1]]
    elseif i == length(x)
        [x[i-1]]
    else
        [x[i-1], x[i+1]]
    end
function neighborhood(m::IsingModel{2, FixedBoundary}, x::AbstractSpinGrid{2}, i)
    i, j = Tuple(CartesianIndices(x)[i])

    ret_i = if i == 1
        [x[i+1, j]]
    elseif i == Base.size(x, 1)
        [x[i-1, j]]
    else
        [x[i-1, j], x[i+1, j]]
    end

    ret_j = if j == 1
        [x[i, j+1]]
    elseif j == Base.size(x, 2)
        [x[i, j-1]]
    else
        [x[i, j-1], x[i, j+1]]
    end

    [ret_i; ret_j]
end

function pdf(m::IsingModel; pfunc=nothing)
    isnothing(pfunc) && (pfunc = Ising.partitionfunc(m))

    x -> Ising.pdf(m, x; pfunc=pfunc)
end
function pdf(m::IsingModel, x; pfunc=nothing)
    isnothing(pfunc) && (pfunc = Ising.partitionfunc(m))

    exp(-m.invtemp*hamiltonian(m, x))/pfunc
end

partitionfunc(m::IsingModel) = sum(spinstates(m)) do spins
    exp(-m.invtemp*hamiltonian(m, spins))
end

MetropolisHastings.currentsample(m::MIsingSampler) = m.world
MetropolisHastings.skip(m::MIsingSampler) = m.skip

MetropolisHastings.log_relprob(m::MIsingSampler, x) = -m.model.invtemp*hamiltonian(m.model, x)
MetropolisHastings.log_trans_prob(m::MIsingSampler, y, x) = 0

MetropolisHastings.log_probdiff(m::MIsingSampler, (ixs, _)) = -m.model.invtemp*hamildiff(m, ixs)
MetropolisHastings.log_trans_probdiff(m::MIsingSampler, dx) = 0

MetropolisHastings.stepto!(m::MIsingSampler, y) = (m.world .= y; nothing)
function MetropolisHastings.stepforward!(rng::Random.AbstractRNG, m::MIsingSampler)
    ixs = rand(rng, eachindex(m.world), m.stepsize)
    flips = rand(rng, Spin, m.stepsize)

    for i in eachindex(ixs)
        m.world[ixs[i]] = xor(m.world[ixs[i]], flips[i])
    end

    (ixs, flips)
end
function MetropolisHastings.stepback!(m::MIsingSampler, (ixs, flips))
    for i in eachindex(ixs)
        m.world[ixs[i]] = xor(m.world[ixs[i]], flips[i])
    end

    nothing
end

end # module Ising
