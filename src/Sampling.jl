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


abstract type MetropolisHastings{T} end
Random.gentype(::Type{MetropolisHastings{T}}) where T = T
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
