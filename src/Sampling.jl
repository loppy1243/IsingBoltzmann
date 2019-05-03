module Sampling
using Random
export MetropolisHastings,
       currentsample,
       step, stepto!, stepforward!, stepback!,
       logrelprob, logprobdiff, logtprob, logtprobdiff,
       log_accept_prob

## Define functions we don't define here so that they can be specialized
for sym in (:currentsample, :stepto!, :stepforward!, :stepback!, :logrelprob, :logprobdiff,
            :logtprob, :logtprobdiff)
    @eval $sym() = throw(MethodError($sym, ()))
end

### Transition probability of x given y
# logtprob(m, x, y)
###
# logtprobdiff(m, dx) =
#   logtprob(m, currentstate(m)-dx, currentstate(m)) #=
#   #= - logtprob(m, currenstate(m), currentstate(m)-dx)

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

log_accept_prob(m::MetropolisHastings, y, x) =
    logrelprob(m, y) - logrelprob(m, x) + logtprob(m, x, y) - logtprob(m, y, x)

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
    copy ? Base.copy(ret) : ret
end
function _rand(rng, m::MetropolisHastings, ::StepType, _)
    x = currentsample(m)
    y = step(m, rng=rng)
    p = log_accept_prob(m, y, x)
    if p >= 0 || log(rand(rng)) <= p
        stepto!(m, y)
        y
    else
        copy(x)
    end
end

end # module Sampling
