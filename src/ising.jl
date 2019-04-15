export MetropolisIsing, dim, hamiltonian

struct MetropolisIsing{D} <: MetropolisHastings
    world::SpinGrid{D}
    coupling::Float64
    invtemp::Float64
    # Flip *at most* this many spins
    stepsize::Int
end
MetropolisIsing(world::SpinGrid, coupling, invtemp, stepsize) =
    MetropolisIsing{ndims(world)}(world, coupling, invtemp, stepsize)
Sampling.StepType(::Type{<:MetropolisIsing}) = Sampling.Reversible()

Base.eltype(T::Type{<:MetropolisIsing}) = SpinGrid{dim(T)}
function Base.show(io::IO, mime::MIME"text/plain", m::MetropolisIsing)
    println(io, "MetropolisIsing(..., $(m.coupling), $(m.invtemp), $(m.stepsize))")
    show(io, mime, m.world)
end

dim(::Type{<:MetropolisIsing{D}}) where D = D
dim(d::MetropolisIsing) = dim(typeof(d))

hamiltonian(m::MetropolisIsing, x) =
    -m.coupling*(
        sum(eachindex(x)) do i; sum(neighborhood(x, i)) do n
            2*xor(x[i], n) - 1
        end end)
## We get a 2 since we have to consider the contribution of the site that flipped and its
## neighbors' contributions
hamildiff(m::MetropolisIsing, (ixs, flips)) =
    -2*m.coupling*(
        sum(ixs) do i; sum(neighborhood(m.world, i)) do n
            # == (2*xor(m.world[i], n) - 1) - (2*xor(flipspin(m.world[i]), n) - 1)
            2*(xor(m.world[i], n) - xor(flipspin(m.world[i]), n))
        end end)

## Can be generalized to a @generated function
neighborhood(x::SpinGrid{1}, i) = (x[i-1], x[i+1])
function neighborhood(x::SpinGrid{2}, I)
    i, j = Tuple(CartesianIndex(I))

    (x[i-1, j], x[i+1, j], x[i, j-1], x[i, j+1])
end

Sampling.currentsample(m::MetropolisIsing) = m.sample

Sampling.logprobdiff(m::MetropolisIsing, dx) = -hamildiff(m, dx)/m.invtemp
Sampling.logtprobdiff(m::MetropolisIsing, dx) = 1

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
