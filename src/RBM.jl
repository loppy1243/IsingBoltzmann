module RBM
## stdlib ######################################################################
using Random, Statistics
################################################################################
## Individual: Internal ########################################################
using ..IsingBoltzmann: bitstrings, @default_first_arg
using LinearAlgebra: mul!
## External ####################################################################
using Random: GLOBAL_RNG
using Reexport: @reexport

export RestrictedBoltzmann,
       biastype, weightstype,
       energy, partitionfunc, pdf, input_pdf, kldiv,
       KLDivGradKernels, update!, train!

struct RestrictedBoltzmann{
        T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, Rf<:Ref{Union{T, Nothing}}
}
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
    partitionfunc::Rf
end
function RestrictedBoltzmann(
        inputbias, hiddenbias, weights;
        learning_rate, cd_num, partitionfunc=nothing
)
    # This function might not be intended for public use...
    T = Base.promote_eltype(inputbias, hiddenbias, weights)

    inputbias, hiddenbias = promote(inputbias, hiddenbias)
    inputbias = convert(AbstractArray{T}, inputbias)
    hiddenbias = convert(AbstractArray{T}, hiddenbias)
    weights = convert(AbstractArray{T}, weights)
    ref = Ref{Union{T, Nothing}}(partitionfunc)

    RestrictedBoltzmann{T, typeof(inputbias), typeof(weights), typeof(ref)}(
        length(inputbias), length(hiddenbias),
        inputbias, hiddenbias, weights, learning_rate, cd_num,
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

    RestrictedBoltzmann(
        inputbias, hiddenbias, weights;
        learning_rate=learning_rate, cd_num=cd_num,
        partitionfunc=partitionfunc
    )
end

Base.eltype(::Type{<:RestrictedBoltzmann{T}}) where T = T

biastype(::Type{<:RestrictedBoltzmann{<:Any, V}}) where V = V
biastype(rbm::RestrictedBoltzmann) = biastype(typeof(rbm))

weightstype(::Type{<:RestrictedBoltzmann{<:Any, <:Any, M}}) where M = M
weightstype(rbm::RestrictedBoltzmann) = weightstype(typeof(rbm))

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
function input_pdf(rbm, inputs)
    exp(-eff_energy(rbm, inputs))/partitionfunc(rbm)
end

sigmoid(x) = inv(one(x)+exp(-x))

condprob_input1(rbm, k, h) = sigmoid(rbm.inputbias[k] + h'rbm.weights[:, k])
condprob_hidden1(rbm, k, σ) = sigmoid(rbm.hiddenbias[k] + rbm.weights[k, :]'σ)

struct AltGibbsSampler{Rbm<:RestrictedBoltzmann} <: Random.Sampler{NTuple{2, BitVector}}
    rbm::Rbm
    inputs::BitVector
    hiddens::BitVector

    AltGibbsSampler(rbm, inputs, hiddens) = new{typeof(rbm)}(rbm, inputs, hiddens)
end
AltGibbsSampler(init::UndefInitializer, rbm) =
    AltGibbsSampler(rbm, BitVector(init, rbm.inputsize), BitVector(init, rbm.hiddensize))
@default_first_arg(
function AltGibbsSampler(rng::AbstractRNG=GLOBAL_RNG, rbm::RestrictedBoltzmann, inputs0)
    hiddens = BitVector(undef, rbm.hiddensize)

    for k in eachindex(hiddens)
        hiddens[k] = rand(rng) <= condprob_hidden1(rbm, k, inputs0)
    end

    AltGibbsSampler(rbm, copy(inputs0), hiddens)
end)
@default_first_arg(
function AltGibbsSampler!(rng::AbstractRNG=GLOBAL_RNG, ag::AltGibbsSampler, inputs0)
    ag.inputs .= inputs0

    for k in eachindex(ag.hiddens)
        ag.hiddens[k] = rand(rng) <= condprob_hidden1(ag.rbm, k, ag.inputs)
    end

    ag
end)

"""
    ag, h = altgibbs([rng=GLOBAL_RNG, ]rbm, inputs0; copy=true)

Returns `ag = AltGibbsSampler(rbm, inputs0)` along with the first sample `h` of
hidden nodes. If `copy` is `false`, then `h` is allowed to alias with internal
state of `ag`.
"""
function altgibbs end
@default_first_arg function altgibbs(
        rng=GLOBAL_RNG, rbm::RestrictedBoltzmann, inputs0; copy=true
)
    ret = AltGibbsSampler(rng, rbm, inputs0)
    ret, copy ? Base.copy(ret.hiddens) : ret.hiddens
end

"""
    h = altgibbs!([rng=GLOBAL_RNG, ]ag::AltGibbsSampler, inputs0; copy=true)

Same as `altgibbs()`, but instead of constructing a new `AltGibbsSampler`, `ag`
is modified in-place.
"""
function altgibbs! end
@default_first_arg function altgibbs!(rng=GLOBAL_RNG, ag::AltGibbsSampler, inputs0; copy=true)
    AltGibbsSampler!(rng, ag, inputs0)
    copy ? Base.copy(ag.hiddens) : ag.hiddens
end

function Random.rand(rng::AbstractRNG, cd::AltGibbsSampler; copy=true)
    for _ = 1:cd.rbm.cd_num
        for k in eachindex(cd.inputs)
            cd.inputs[k] = rand(rng) <= condprob_input1(cd.rbm, k, cd.hiddens)
        end
        for k in eachindex(cd.hiddens)
            cd.hiddens[k] = rand(rng) <= condprob_hidden1(cd.rbm, k, cd.inputs)
        end
    end

    copy ? (Base.copy(cd.inputs), Base.copy(cd.hiddens)) : (cd.inputs, cd.hiddens)
end
function Random.rand!(rng::AbstractRNG, (inputs, hiddens), cd::AltGibbsSampler)
    for _ = 1:cd.rbm.cd_num
        for k in eachindex(cd.inputs)
            inputs[k] = cd.inputs[k] = rand(rng) <= condprob_input1(cd.rbm, k, cd.hiddens)
        end
        for k in eachindex(cd.hiddens)
            hiddens[k] = cd.hiddens[k] = rand(rng) <= condprob_input1(cd.rbm, k, cd.hiddens)
        end
    end

    (inputs, hiddens)
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

module KLDivGradKernels
    using ..RBM
    using ..RBM: AltGibbsSampler
    export KLDivGradKernel, ExactKernel, ApproxKernel, CuArrayKernel

    ## Subtypes must implement the method
    ##     σgrad, hgrad, Wgrad = (::KLDivGradKernel)(rng, rbm, data)
    abstract type KLDivGradKernel end

    Base.@kwdef struct PosNegNodes
        pos_inputs::Vector{BitVector}
        pos_hiddens::Vector{BitVector}
        neg_inputs::Vector{BitVector}
        neg_hiddens::Vector{BitVector}
    end
    function PosNegNodes(rbm, batchsize)
        PosNegNodes(
            [BitVector(undef, rbm.inputsize)  for _ = 1:batchsize],
            [BitVector(undef, rbm.hiddensize) for _ = 1:batchsize],
            [BitVector(undef, rbm.inputsize)  for _ = 1:batchsize],
            [BitVector(undef, rbm.hiddensize) for _ = 1:batchsize]
        )
    end

    Base.@kwdef struct Grad{V, M}
        inputs::V
        hiddens::V
        weights::M
    end
    Grad(rbm) = Grad{biastype(rbm), weightstype(rbm)}(
        biastype(rbm)(undef, rbm.inputsize), biastype(rbm)(undef, rbm.hiddensize),
        weightstype(rbm)(undef, rbm.hiddensize, rbm.inputsize)
    )

    ## See doc/kldiv_grad.tex for the difference between ExactKernel and ApproxKernel.
    struct ExactKernel{
            T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, Rbm<:RestrictedBoltzmann{T, V, M}
    } <: KLDivGradKernel
        σh_sampler::AltGibbsSampler{Rbm}
        grad::Grad{V, M}
        pos_h::BitVector
        condavg_h::Vector{Float64}
        hσ_prod::Matrix{Float64}
    end
    ExactKernel(rbm) = ExactKernel{eltype(rbm), biastype(rbm), weightstype(rbm), typeof(rbm)}(
        AltGibbsSampler(undef, rbm),
        Grad(rbm),
        BitVector(undef, rbm.hiddensize),
        Vector{Float64}(undef, rbm.hiddensize),
        Matrix{Float64}(undef, rbm.hiddensize, rbm.inputsize)
    )

    struct ApproxKernel{
            T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, Rbm<:RestrictedBoltzmann{T, V, M}
    } <: KLDivGradKernel
        σh_sampler::AltGibbsSampler{Rbm}
        grad::Grad{V, M}
        pos_h::BitVector
        hσ_prod::Matrix{Float64}
    end
    ApproxKernel(rbm) = ApproxKernel{eltype(rbm), biastype(rbm), weightstype(rbm), typeof(rbm)}(
        AltGibbsSampler(undef, rbm),
        Grad(rbm),
        BitVector(undef, rbm.hiddensize),
        Matrix{Float64}(undef, rbm.hiddensize, rbm.inputsize)
    )

    struct CuArrayKernel <: KLDivGradKernel
    end
end
@reexport using .KLDivGradKernels

condavg_hidden!(out, rbm, σ) = out .= sigmoid.(rbm.hiddenbias .+ mul!(out, rbm.weights, σ))
## Exact hidden conditional mean.
function (kern::ExactKernel)(rng, rbm, batch)
    z = zero(eltype(rbm))
    fill!(kern.grad.inputs, z)
    fill!(kern.grad.hiddens, z)
    fill!(kern.grad.weights, z)

    for σ⁺ in batch
        ## NOTE the order of operations on kern.condavg_h in particular. As it stands, this is
        ## the correct order
        
        altgibbs!(rng, kern.σh_sampler, σ⁺; copy=false)
        σ⁻, h⁻ = rand(rng, kern.σh_sampler; copy=false)

        kern.grad.inputs .+= σ⁻ .- σ⁺

        kern.grad.hiddens .+= condavg_hidden!(kern.condavg_h, rbm, σ⁻)
        kern.grad.hiddens .-= condavg_hidden!(kern.condavg_h, rbm, σ⁺)

        kern.grad.weights .+= mul!(kern.hσ_prod, h⁻, σ⁻')
        kern.grad.weights .-= mul!(kern.hσ_prod, kern.condavg_h, σ⁺')
    end
    L = length(batch)
    kern.grad.inputs  ./= L
    kern.grad.hiddens ./= L
    kern.grad.weights ./= L

    kern.grad.inputs, kern.grad.hiddens, kern.grad.weights
end

## Approximate hidden conditional mean.
function (kern::ApproxKernel)(rng, rbm, batch)
    z = zero(eltype(rbm))
    fill!(kern.grad.inputs, z)
    fill!(kern.grad.hiddens, z)
    fill!(kern.grad.weights, z)

    for σ⁺ in batch
        kern.pos_h .= altgibbs!(rng, kern.σh_sampler, σ⁺; copy=false)
        σ⁻, h⁻ = rand(rng, kern.σh_sampler; copy=false)

        kern.grad.inputs .+= σ⁻ .- σ⁺
        kern.grad.hiddens .+= h⁻ .- kern.pos_h
        kern.grad.weights .+= mul!(kern.hσ_prod, h⁻, σ⁻')
        kern.grad.weights .-= mul!(kern.hσ_prod, kern.pos_h, σ⁺')
    end
    L = length(batch)
    kern.grad.inputs  ./= L
    kern.grad.hiddens ./= L
    kern.grad.weights ./= L

    kern.grad.inputs, kern.grad.hiddens, kern.grad.weights
end

@default_first_arg function update!(rng::AbstractRNG=GLOBAL_RNG, rbm, kern, batch)
    σgrad, hgrad, Wgrad = kern(rng, rbm, batch)

    rbm.inputbias  .-= rbm.learning_rate.*σgrad
    rbm.hiddenbias .-= rbm.learning_rate.*hgrad
    rbm.weights    .-= rbm.learning_rate.*Wgrad
    rbm.partitionfunc[] = nothing

    rbm
end

@default_first_arg function train!(rng::AbstractRNG=GLOBAL_RNG, rbm, kern, minibatches)
    perm = randperm(rng, size(minibatches, ndims(minibatches)))
    permute!(minibatches, perm)

    for b in minibatches
        update!(rng, rbm, kern, b)
    end

    perm
end

end # module RBM
