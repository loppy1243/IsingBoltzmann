module RBM
## stdlib ######################################################################
using Random, Statistics
################################################################################
## Individual: Internal ########################################################
using ..IsingBoltzmann: bitstrings, @default_first_arg
using LinearAlgebra: mul!
## External ####################################################################
using Random: GLOBAL_RNG

export RestrictedBoltzmann, energy, partitionfunc, pdf, input_pdf, update!, train!, kldiv

const MFloat64 = Union{Nothing, Float64}
struct RestrictedBoltzmann{T, V, M, Rf<:Ref{MFloat64}}
    inputsize::Int
    hiddensize::Int
    
    ## Parameters
    inputbias::V
    hiddenbias::V
    weights::M

    ## Hyperparameters
    learning_rate::Float64
    cd_num::Int

    ## Scratch space
    inputgrad::Vector{T}
    hiddengrad::Vector{T}
    weightsgrad::Matrix{T}

    condavg_hidden::Vector{T}
    hiddeninput_prod::Matrix{T}

    partitionfunc::Rf
end
function RestrictedBoltzmann(
        inputbias, hiddenbias, weights;
        learning_rate, cd_num, partitionfunc=nothing
)
    ref = Ref{MFloat64}(partitionfunc)
    RestrictedBoltzmann{eltype(inputbias), typeof(inputbias), typeof(weights), typeof(ref)}(
        length(inputbias), length(hiddenbias),
        inputbias, hiddenbias, weights, learning_rate, cd_num,
        similar(inputbias), similar(hiddenbias), similar(weights),
        similar(hiddenbias), similar(weights),
        ref
    )
end
function RestrictedBoltzmann(
        inputsize, hiddensize;
        init=zeros, inputbias_init=nothing, hiddenbias_init=nothing, weights_init=nothing,
        learning_rate, cd_num,
        partitionfunc=nothing
)
    inputbias = if isnothing(inputbias_init)
        init((inputsize,)) else inputbias_init((inputsize,))
    end
    inputbias = if isnothing(inputbias_init)
        init((inputsize,)) else inputbias_init((inputsize,))
    end
    hiddenbias = if isnothing(hiddenbias_init)
        init((hiddensize,)) else hiddenbias_init((hiddensize,))
    end
    weights = if isnothing(weights_init)
        init((hiddensize, inputsize)) else weights_init((hiddensize, inputsize))
    end

    ref = Ref{MFloat64}(partitionfunc)

    RestrictedBoltzmann{eltype(inputbias), typeof(inputbias), typeof(weights), typeof(ref)}(
        inputsize, hiddensize,
        inputbias, hiddenbias, weights, learning_rate, cd_num,
        similar(inputbias), similar(hiddenbias), similar(weights),
        similar(hiddenbias), similar(weights),
        ref
    )
end

Base.eltype(rbm::RestrictedBoltzmann{T}) where T = T

energy(rbm, inputs, hiddens) =
    -sum(rbm.inputbias.*inputs) - sum(hiddens.*rbm.hiddenbias) - sum(hiddens.*rbm.weights*inputs)
eff_energy(rbm, inputs) = -rbm.inputbias'inputs - (
    sum(eachindex(rbm.hiddenbias)) do i
        log(one(eltype(rbm)) + exp(rbm.hiddenbias[i] + rbm.weights[i, :]'inputs))
    end
)

partitionfunc(rbm) = isnothing(rbm.partitionfunc[]) ? _partitionfunc(rbm) : rbm.partitionfunc[]
_partitionfunc(rbm) = rbm.partitionfunc[] = sum(
    exp(-energy(rbm, inputs, hiddens))
    for hiddens in bitstrings(rbm.hiddensize),
        inputs  in bitstrings(rbm.inputsize)
)

pdf(rbm) = (inputs, hiddens) -> RBM.pdf(rbm, inputs, hiddens)
pdf(rbm, inputs, hiddens) = exp(-energy(rbm, inputs, hiddens))/partitionfunc(rbm)
input_pdf(rbm) = inputs -> RBM.input_pdf(rbm, inputs)
input_pdf(rbm, inputs) = exp(-eff_energy(rbm, inputs))/partitionfunc(rbm)

sigmoid(x) = inv(one(x)+exp(-x))

condprob_input1(rbm, k, h) = sigmoid(rbm.inputbias[k] + h'rbm.weights[:, k])
condprob_hidden1(rbm, k, σ) = sigmoid(rbm.hiddenbias[k] + rbm.weights[k, :]'σ)

function condavg_hidden!(rbm, σ)
    mul!(rbm.condavg_hidden, rbm.weights, σ)
    rbm.condavg_hidden .= sigmoid.(rbm.hiddenbias .+ rbm.condavg_hidden)
end

struct AltGibbsSampler <: Random.Sampler{NTuple{2, BitVector}}
    rbm::RestrictedBoltzmann
    inputs::BitVector
    hiddens::BitVector
end
@default_first_arg(
function AltGibbsSampler(rng::AbstractRNG=GLOBAL_RNG, rbm, inputs0)
    hiddens = BitVector(undef, rbm.hiddensize)

    for k in eachindex(hiddens)
        hiddens[k] = rand(rng) <= condprob_hidden1(rbm, k, inputs0)
    end

    AltGibbsSampler(rbm, copy(inputs0), hiddens)
end)

"""
    ag, h = altgibbs([rng=GLOBAL_RNG, ]rbm, inputs0)

Return `ag = AltGibbsSampler(rbm, inputs0)` along with the first sample `h` of
hidden nodes.
"""
function altgibbs end
@default_first_arg function altgibbs(rng=GLOBAL_RNG, rbm, inputs0)
    ret = AltGibbsSampler(rng, rbm, inputs0)
    ret, copy(ret.hiddens)
end

function Random.rand(rng::AbstractRNG, cd::AltGibbsSampler; copy=true)
    for k in eachindex(cd.inputs)
        cd.inputs[k] = rand(rng) <= condprob_input1(cd.rbm, k, cd.hiddens)
    end
    for k in eachindex(cd.hiddens)
        cd.hiddens[k] = rand(rng) <= condprob_hidden1(cd.rbm, k, cd.inputs)
    end

    copy ? (Base.copy(cd.hiddens), Base.copy(cd.inputs)) : (cd.hiddens, cd.inputs)
end

function entropy(batch)
    counts = Dict{eltype(batch), Int}()
    for σ in batch
        counts[σ] = get(counts, σ, 0) + 1
    end

    -sum(keys(counts)) do σ
        counts[σ]/length(batch)*log(counts[σ]/length(batch))
    end
end

## Exact
kldiv(rbm, target_pdf::Function) = sum(bitstrings(rbm.inputsize)) do inputs
    p = target_pdf(inputs)
    p*log(p/input_pdf(rbm, inputs))
end
## Approximation
function kldiv(rbm, batch)
    L = length(batch)

    avg_log_likelihood = mean(batch) do σ
        log(input_pdf(rbm, σ))
    end

    entrop = entropy(batch)

    -avg_log_likelihood - entrop
end

## There are two potential ways this should maybe be implemented. See doc/kldiv_grad.tex for
## details.
@default_first_arg function kldiv_grad!(rng=GLOBAL_RNG, rbm, batch)
    z = zero(eltype(rbm))
    rbm.inputgrad   .= z
    rbm.hiddengrad  .= z
    rbm.weightsgrad .= z

    L = length(batch)
    for σ in batch
        ## Method 1
        σh_sampler = AltGibbsSampler(rng, rbm, σ)
        h2, σ2 = rand(rng, σh_sampler; copy=false)

        condavg_hidden!(rbm, σ2)
        rbm.inputgrad   .+= σ2
        rbm.hiddengrad  .+= rbm.condavg_hidden
        mul!(rbm.hiddeninput_prod, h2, σ2')
        rbm.weightsgrad .+= rbm.hiddeninput_prod

        condavg_hidden!(rbm, σ)
        rbm.inputgrad   .-= σ
        rbm.hiddengrad  .-= rbm.condavg_hidden
        mul!(rbm.hiddeninput_prod, rbm.condavg_hidden, σ')
        rbm.weightsgrad .-= rbm.hiddeninput_prod

#        ## Method 2
#        σh_sampler, h = altgibbs(rbm, σ)
#
#        h2, σ2 = rand(σh_sampler; copy=false)
#        rbm.inputgrad   .+= σ2 .- σ
#        rbm.hiddengrad  .+= h2 .- h
#        mul!(rbm.hiddeninput_prod, h2, σ2')
#        rbm.weightsgrad .+= rbm.hiddeninput_prod
#        mul!(rbm.hiddeninput_prd, h, σ')
#        rbm.weightsgrad .-= rbm.hiddeninput_prod
    end
    rbm.inputgrad   ./= L
    rbm.hiddengrad  ./= L
    rbm.weightsgrad ./= L

    rbm.inputgrad, rbm.hiddengrad, rbm.weightsgrad
end

@default_first_arg function update!(rng=GLOBAL_RNG, rbm, batch)
    kldiv_grad!(rbm, batch)

    rbm.inputbias  .-= rbm.learning_rate.*rbm.inputgrad
    rbm.hiddenbias .-= rbm.learning_rate.*rbm.hiddengrad
    rbm.weights    .-= rbm.learning_rate.*rbm.weightsgrad
    rbm.partitionfunc[] = nothing

    rbm
end

@default_first_arg function train!(rng=GLOBAL_RNG, rbm, minibatches)
    perm = randperm(rng, size(minibatches, ndims(minibatches)))
    permute!(minibatches, perm)

    for (k, b) in enumerate(minibatches)
        update!(rng, rbm, b)
    end

    perm
end

end # module RBM
