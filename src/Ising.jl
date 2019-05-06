module Ising
using ..Sampling, ..Spins, ..PeriodicArrays
using ..IsingBoltzmann: cartesian_prod
export MetropolisIsing, dim, hamiltonian, partitionfunc

# D = Dimensions, B = BoundaryCondition
mutable struct IsingModel{D, B}
    size::Dims{D}
    coupling::Float64
    invtemp::Float64
    partitionfunc::Union{Nothing, Float64}

    IsingModel{D, B}(sz::Dims{D}; coupling, invtemp) where {D, B} =
        new(sz, coupling, invtemp, nothing)
    IsingModel{D, B}(sz::Int...; coupling, invtemp) where {D, B} =
        new(sz, coupling, invtemp, nothing)
end
boundarycond(::Type{<:IsingModel{<:Any, B}}) where B = B
boundarycond(m::IsingModel) = boundarycond(typeof(m))

abstract type BoundaryCondition end
struct FixedBoundary <: BoundaryCondition end
struct PeriodicBoundary <: BoundaryCondition end

const AbstractSpinGrid{D} = AbstractArray{Spin, D}
struct MetropolisIsing{D, I<:IsingModel{D}, W<:AbstractSpinGrid{D}} <: MetropolisHastings{W}
    model::I
    world::W

    # Flip *at most* this many spins
    stepsize::Int
    skip::Int

    function MetropolisIsing{D, I, W}(model::I, world::W; stepsize=1, skip=0) where
                            {D, I<:IsingModel{D}, W<:AbstractSpinGrid{D}}
        @assert model.size == Base.size(world)

        new(model, world, stepsize, skip)
    end
end
MetropolisIsing(model::IsingModel{D}, world::AbstractSpinGrid{D}; stepsize=1, skip=0) where D =
    MetropolisIsing{D, typeof(model), typeof(world)}(model, world; stepsize=stepsize, skip=skip)
Sampling.StepType(::Type{<:MetropolisIsing}) = Sampling.Reversible()

function Base.show(io::IO, mime::MIME"text/plain", m::MetropolisIsing)
    println(io,
        "MetropolisIsing(..., IsingModel($(m.model.coupling), $(m.model.invtemp)), ",
        "$(m.stepsize), $(m.skip))"
    )
    show(io, mime, m.world)
end

worldtype(::Type{<:MetropolisIsing{<:Any, <:Any, W}}) where W = W
worldtype(m::MetropolisIsing) = worldtype(typeof(m))

## Define on Base.ndims?
ndims(::Type{<:IsingModel{D}}) where D = D
ndims(::Type{<:MetropolisIsing{D}}) where D = D
ndims(d) = ndims(typeof(d))
ndims(T::Type) = throw(MethodError(ndims, T))

## Define on Base.size?
size(m::IsingModel) = m.size
size(m::MetropolisIsing) = size(m.model)

nspins(m) = prod(Ising.size(m))

hamiltonian(m::MetropolisIsing) = hamiltonian(m.model, m.world)
hamiltonian(m::IsingModel, x) =
    -m.coupling*(
        sum(eachindex(x)) do i; sum(neighborhood(m, x, i)) do n
            1 - 2*xor(x[i], n)
        end end
    )
## We get a 2 since we have to consider the contribution of the site that flipped and its
## neighbors' contributions
hamildiff(m::MetropolisIsing, ixs) = hamildiff(m.model, m.world, xs)
hamildiff(m::IsingModel, x, ixs) =
    -2*m.coupling*(
        sum(ixs) do i; sum(neighborhood(m, x, i)) do n
            # == (2*xor(x[i], n) - 1) - (2*xor(flipspin(x[i]), n) - 1)
            2*(xor(flipspin(x[i]), n) - xor(x[i], n))
        end end)

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

pdf(m::IsingModel) = x -> Ising.pdf(m, x)
pdf(m::IsingModel, x) = exp(-m.invtemp*hamiltonian(m, x))/partitionfunc(m)
partitionfunc(m::IsingModel) =
    isnothing(m.partitionfunc) ? _partitionfunc(m) : m.partitionfunc
function _partitionfunc(m)
    sum = 0.0
    for spins in spinstrings(nspins(m))
        sum += exp(-m.invtemp*hamiltonian(m, reshape(spins, m.size)))
    end

    m.partitionfunc = sum

    sum
end

#exact_rand(m::MetropolisIsing) = exact_rand(GLOBAL_RNG, m)
#function exact_rand(rng, m::MetropolisIsing)
#    spingrid = similar(m.world)
#    sum = 0.0
#    for spins in cartesian_prod(SPINS, length(spins))
#        spingrid[:] .= spins
#        spingrid == x && break
#
#        sum += pdf(m, args)
#
#        rand(rng) <= sum && return spingrid
#    end
#
#    world
#end

Sampling.currentsample(m::MetropolisIsing) = m.world
Sampling.skip(m::MetropolisIsing) = m.skip

Sampling.log_relprob(m::MetropolisIsing, x) = -m.invtemp*hamiltonian(m, x)
Sampling.log_trans_prob(m::MetropolisIsing, y, x) = 0

Sampling.log_probdiff(m::MetropolisIsing, (ixs, _)) = -m.invtemp*hamildiff(m, ixs)
Sampling.log_trans_probdiff(m::MetropolisIsing, dx) = 0

Sampling.stepto!(m::MetropolisIsing, y) = (m.world .= y; nothing)
function Sampling.stepforward!(rng::Random.AbstractRNG, m::MetropolisIsing)
    ixs = rand(rng, eachindex(m.world), m.stepsize)
    flips = rand(rng, Spin, m.stepsize)

    for i in eachindex(ixs)
        m.world[ixs[i]] = xor(m.world[ixs[i]], flips[i])
    end

    (ixs, flips)
end
function Sampling.stepback!(m::MetropolisIsing, (ixs, flips))
    for i in eachindex(ixs)
        m.world[ixs[i]] = xor(m.world[ixs[i]], flips[i])
    end

    nothing
end

end # module Ising
