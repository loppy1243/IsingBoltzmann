module Sampling
using Random
export MetropolisHastings,
       currentsample,
       step, stepto!, stepforward!, stepback!,
       log_relprob, log_probdiff, log_trans_prob, log_trans_probdiff,
       log_accept_prob

## Define functions we don't define here so that they can be specialized
for sym in (:currentsample, :stepto!, :stepback!, :log_probdiff, :log_trans_prob,
            :log_trans_probdiff)
    @eval $sym() = throw(MethodError($sym, ()))
end

### Transition probability of y given x
# log_trans_prob(m, y, x)
###
# log_trans_probdiff(m, dx) =
#   log_trans_prob(m, currentstate(m)-dx, currentstate(m)) #=
#   #= - log_trans_prob(m, currenstate(m), currentstate(m)-dx)

"""
    MetropolisHastings{T}

Abstract type for Metropolis-Hastings samplers producing elements of type `T`.

There are two main sets of methods to implement depending on the trait `Sampling.StepType`.
See `Sampling.Default` and `Sampling.Reversible`.
"""
abstract type MetropolisHastings{T} end
Random.gentype(::Type{MetropolisHastings{T}}) where T = T
skip(::MetropolisHastings) = 0

"""
    Sampling.StepType

Trait specifying how `MetropolisHastings` steps can be performed.
"""
abstract type StepType end

"""
     Sampling.StepType(::MetropolisHastings) = Sampling.Default()

Default step type for `MetropolisHastings`. Each Metropolis step generates a completely new
state. Methods to implement:

    step(::AbstractRNG, ::MetropolisHastings)
    stepto!(::MetropolisHastings, x)
    log_relprob(::MetropolisHastings, x)
    log_trans_prob(::MetropolisHastings, y, x)

    # Optional
    log_relprob(::MetropolisHastings, y, x)
    skip(::MetropolisHastings)
"""
struct Default <: StepType end
"""
     Sampling.StepType(::MetropolisHastings) = Sampling.Reversible()

Reversible step type for `MetropolisHastings`. Implement this if Metropolis steps are
reversible and can be performed as a small "delta" between states. Methods to implement:

    stepforward!(::AbstractRNG, ::MetropolisHastings)
    stepback!(::MetropolisHastings, dx)
    log_probdiff(::MetropolisHastings, dx)
    log_trans_probdiff(::MetropolisHastings, dx)

    # Optional
    skip(::MetropolisHastings)
"""
struct Reversible <: StepType end
StepType(::Type{<:MetropolisHastings}) = Default()
StepType(m::MetropolisHastings) = StepType(typeof(m))

step(rng::AbstractRNG, m::MetropolisHastings) = _step(rng, m, StepType(m))
function _step(rng, m::MetropolisHastings, ::Reversible)
    y, dx = stepforward!(rng, m)
    y_copy = copy(y)
    stepback!(m, dx)

    y_copy
end

step(m::MetropolisHastings) = step(Random.GLOBAL_RNG, m)
stepforward!(m::MetropolisHastings) = stepforward!(Random.GLOBAL_RNG, m)

log_accept_prob(m::MetropolisHastings, y, x) =
    log_relprob(m, y, x) + log_trans_prob(m, x, y) - log_trans_prob(m, y, x)

log_relprob(::MetropolisHastings, y, x) = log_relprob(m, y) - log_relprob(m, x)

Random.rand(rng::AbstractRNG, m::Random.SamplerTrivial{<:MetropolisHastings}; copy=true) =
    _rand(rng, m[], StepType(m), copy)
function _rand(rng, m::MetropolisHastings, ::Reversible, copy)
    for _ = 0:skip(m)
        dx = stepforward!(rng, m)
        p = log_probdiff(m, dx) + log_trans_probdiff(m, dx)

        if !(p >= 0 || log(rand(rng)) <= p)
            stepback!(m, dx)
        end
    end
    copy ? Base.copy(currentsample(m)) : currentsample(m)
end
function _rand(rng, m::MetropolisHastings, ::StepType, copy)
    for _ = 0:skip(m)
        x = currentsample(m)
        y = step(rng, m)
        p = log_accept_prob(m, y, x)

        if p >= 0 || log(rand(rng)) <= p
            stepto!(m, y)
        else
    end

    copy ? Base.copy(currentsample(m)) : currentsample(m)
end

end # module Sampling
