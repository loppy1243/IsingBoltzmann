module MetropolisHastings
using Random
export MetropolisHastingsSampler,
       currentsample,
       step, stepto!, stepforward!, stepback!,
       log_relprob, log_probdiff, log_trans_prob, log_trans_probdiff,
       log_accept_prob

## Define functions we don't define here so that they can be specialized
for sym in (:currentsample, :stepto!, :stepback!, :log_relprob, :log_probdiff,
            :log_trans_prob, :log_trans_probdiff)
    @eval function $sym end
end

### Transition probability of y given x
# log_trans_prob(m, y, x)
###
# log_trans_probdiff(m, dx) =
#   log_trans_prob(m, currentstate(m)-dx, currentstate(m)) #=
#   #= - log_trans_prob(m, currenstate(m), currentstate(m)-dx)

"""
    MetropolisHastingsSampler{T}

Abstract type for Metropolis-Hastings samplers producing elements of type `T`.

There are two main sets of methods to implement depending on the trait `Sampling.StepType`.
See `Sampling.Default` and `Sampling.Reversible`.
"""
abstract type MetropolisHastingsSampler{T} <: Random.Sampler{T} end
const MHSampler = MetropolisHastingsSampler
skip(::MHSampler) = 0

"""
    MetropolisHastings.StepType

Trait specifying how Metropolis-Hastings steps can be performed.
"""
abstract type StepType end

"""
     MetropolisHastings.StepType(sampler) = Sampling.Default()

Default step type. Each Metropolis step generates a completely new state. Methods to
implement for type `T<:MetropolisHastingsSampler`:

    step(::Random.AbstractRNG, ::T)
    stepto!(::T, x)
    log_relprob(::T, x)
    log_trans_prob(::T, y, x)

    # Optional
    skip(::T)
"""
struct Default <: StepType end
"""
     Sampling.StepType(sampler) = Sampling.Reversible()

Reversible step type. Implement this if Metropolis steps are reversible and can be performed
as a small "delta" between states. Methods to implement for type
`T<:MetropolisHastingsSampler`:

    stepforward!(::Random.AbstractRNG, ::T)
    stepback!(::T, dx)
    log_probdiff(::T, dx)
    log_trans_probdiff(::T, dx)

    # Optional
    skip(::T)

    # Needed for log_accept_prob()
    log_relprob(::T, x)
    log_trans_prob(::T, y, x)
"""
struct Reversible <: StepType end
StepType(::Type{<:MHSampler}) = Default()
StepType(m::MHSampler) = StepType(typeof(m))

step(rng::Random.AbstractRNG, m::MHSampler) = _step(rng, m, StepType(m))
function _step(rng, m::MHSampler, ::Reversible)
    dx = stepforward!(rng, m)
    ret = copy(currentsample(m))
    stepback!(m, dx)

    ret
end

step(m::MHSampler) = step(Random.GLOBAL_RNG, m)
stepforward!(m::MHSampler) = stepforward!(Random.GLOBAL_RNG, m)

log_accept_prob(m::MHSampler, y, x) =
    log_relprob(m, y) - log_relprob(m, x) + log_trans_prob(m, x, y) - log_trans_prob(m, y, x)

Random.rand(rng::Random.AbstractRNG, m::MHSampler; copy=true) =
    _rand(rng, m, StepType(m), copy)
function _rand(rng, m::MHSampler, ::Reversible, copy)
    for _ = 0:skip(m)
        dx = stepforward!(rng, m)
        p = log_probdiff(m, dx) + log_trans_probdiff(m, dx)

        if !(p >= 0 || log(rand(rng)) <= p)
            stepback!(m, dx)
        end
    end
    copy ? Base.copy(currentsample(m)) : currentsample(m)
end
function _rand(rng, m::MHSampler, ::StepType, copy)
    for _ = 0:skip(m)
        x = currentsample(m)
        y = step(rng, m)
        p = log_accept_prob(m, y, x)

        if p >= 0 || log(rand(rng)) <= p
            stepto!(m, y)
        end
    end

    copy ? Base.copy(currentsample(m)) : currentsample(m)
end

end # module Sampling
