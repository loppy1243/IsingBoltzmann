module Ising
using ..Sampling, ..Spins, ..PeriodicArrays
using ..IsingBoltzmann: cartesian_prod
export MetropolisIsing, dim, hamiltonian, partitionfunc

const MFloat64 = Union{Nothing, Float64}
struct MetropolisIsing{D, R<:Ref{MFloat64}} <: MetropolisHastings{SpinGrid{D}}
    world::SpinGrid{D}
    coupling::Float64
    invtemp::Float64
    # Flip *at most* this many spins
    stepsize::Int
    skip::Int

    partitionfunc::R

    MetropolisIsing{D, R}(world, coupling, invtemp, stepsize, skip, partitionfunc=R(nothing)) #=
         =# where {D, R<:Ref{MFloat64}} =
        new(world, coupling, invtemp, stepsize, skip, partitionfunc)
end
function MetropolisIsing(world, coupling, invtemp, stepsize, skip, partitionfunc=nothing)
    ref = Ref{MFloat64}(partitionfunc)
    MetropolisIsing{ndims(world), typeof(ref)}(world, coupling, invtemp, stepsize, skip, ref)
end
Sampling.StepType(::Type{<:MetropolisIsing}) = Sampling.Reversible()

function Base.show(io::IO, mime::MIME"text/plain", m::MetropolisIsing)
    println(io, "MetropolisIsing(..., $(m.coupling), $(m.invtemp), $(m.stepsize))")
    show(io, mime, m.world)
end

dim(::Type{<:MetropolisIsing{D}}) where D = D
dim(d::MetropolisIsing) = dim(typeof(d))

hamiltonian(m::MetropolisIsing) = hamiltonian(m, m.world)
hamiltonian(m::MetropolisIsing, x) =
    -m.coupling*(
        sum(eachindex(x)) do i; sum(neighborhood(x, i)) do n
            1 - 2*xor(x[i], n)
        end end
    )
## We get a 2 since we have to consider the contribution of the site that flipped and its
## neighbors' contributions
hamildiff(m::MetropolisIsing, (ixs, flips)) =
    -2*m.coupling*(
        sum(ixs) do i; sum(neighborhood(m.world, i)) do n
            # == (2*xor(m.world[i], n) - 1) - (2*xor(flipspin(m.world[i]), n) - 1)
            2*(xor(flipspin(m.world[i]), n) - xor(m.world[i], n))
        end end)

## Can be generalized to a @generated function
neighborhood(x::PeriodicArray{<:Any, 1}, i) = (x[i-1], x[i+1])
function neighborhood(x::PeriodicArray{<:Any, 2}, I)
    i, j = Tuple(CartesianIndex(I))

    (x[i-1, j], x[i+1, j], x[i, j-1], x[i, j+1])
end
neighborhood(x::AbstractArray{<:Any, 1}, i) =
    if i == 1
        (x[i+1],)
    elseif i == length(x)
        (x[i-1],)
    else
        (x[i-1], x[i+1])
    end

pdf(m::MetropolisIsing, x) = exp(-m.invtemp*hamiltonian(m, x))/partitionfunc(m)
partitionfunc(m::MetropolisIsing) =
    isnothing(m.partitionfunc[]) ? _partitionfunc(m) : m.partitionfunc[]
function _partitionfunc(m)
    spingrid = similar(m.world)
    sum = 0.0
    for spins in cartesian_prod((SPINDN, SPINUP), length(spingrid))
        spingrid[:] .= spins
        sum += exp(-m.invtemp*hamiltonian(m, spingrid))
    end

    m.partitionfunc[] = sum

    sum
end

exact_rand(m::MetropolisIsing) = exact_rand(GLOBAL_RNG, m)
function exact_rand(rng, m::MetropolisIsing)
    spingrid = similar(m.world)
    sum = 0.0
    for spins in cartesian_prod(SPINS, length(spins))
        spingrid[:] .= spins
        spingrid == x && break

        sum += pdf(m, args)

        rand(rng) <= sum && return spingrid
    end

    world
end

Sampling.currentsample(m::MetropolisIsing) = m.sample
Sampling.skip(m::MetropolisIsing) = m.skip

Sampling.logrelprob(m::MetropolisIsing, x) = -m.invtemp*hamiltonian(m, x)
Sampling.logtprob(m::MetropolisIsing, y, x) = 0

Sampling.logprobdiff(m::MetropolisIsing, dx) = -m.invtemp*hamildiff(m, dx)
Sampling.logtprobdiff(m::MetropolisIsing, dx) = 0

Sampling.stepto!(m::MetropolisIsing, y) = (m.world .= y; nothing)
function Sampling.stepforward!(m::MetropolisIsing; rng=GLOBAL_RNG)
    ixs = rand(eachindex(m.world), m.stepsize)
    flips = rand(Spin, m.stepsize)

    for i in eachindex(ixs)
        m.world[ixs[i]] = xor(m.world[ixs[i]], flips[i])
    end

    m.world, (ixs, flips)
end
function Sampling.stepback!(m::MetropolisIsing, (ixs, flips))
    for i in eachindex(ixs)
        m.world[ixs[i]] = xor(m.world[ixs[i]], flips[i])
    end

    nothing
end

end # module Ising
