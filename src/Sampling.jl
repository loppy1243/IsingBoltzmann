module Sampling
using Random
export MetropolisHastings,
       currentsample,
       step, stepto!, stepforward!, stepback!,
       logrelprob, logprobdiff, logtprob, logtprobdiff

## Define functions we don't define here so that they can be specialized
for sym in (:currentsample, :stepto!, :stepforward!, :stepback!, :logrelprob, :logprobdiff,
            :logtprob, :logtprobdiff)
    @eval $sym() = throw(MethodError($sym, ()))
end

abstract type MetropolisHastings{T} <: Random.Sampler{T} end

abstract type StepType end
struct Default <: StepType end
struct Reversible <: StepType end
StepType(::Type{<:MetropolisHastings}) = Default()
StepType(m::MetropolisHastings) = StepType(typeof(m))

step(m::MetropolisHastings, rng=GLOBAL_RNG) = _step(m::MetropolisHastings, StepType(m), rng)
function _step(m::MetropolisHastings, ::Reversible, rng)
    y, dx = stepforward!(m, rng=rng)
    y_copy = copy(y)
    stepback!(m, dx)

    y_copy
end

Random.rand(rng::AbstractRNG, m::MetropolisHastings; copy=true) =
    _rand(rng, m, StepType(m), copy)
function _rand(rng, m::MetropolisHastings, ::Reversible, copy)
    y, dx = stepforward!(m, rng=rng)
    p = logprobdiff(m, dx) + logtprobdiff(m, dx)

    ret = if p >= 0 || log(rand(rng)) <= p
        y
    else
        stepback!(m, dx)
        m.world
    end
    copy ? copy(ret) : ret
end
function _rand(rng, m::MetropolisHastings, ::StepType, _)
    x = currentsample(m)
    y = step(m, rng=rng)
    p = logrelprob(m, y) - logrelprob(m, x) + logtprob(m, x, y) - logtprob(m, y, x)
    if p >= 0 || log(rand(rng)) <= p
        stepto!(m, y)
        y
    else
        copy(x)
    end
end

end # module Sampling
