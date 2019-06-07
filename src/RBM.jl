module RBM
## stdlib ######################################################################
using Random, Statistics
## External ####################################################################
import Flux.Optimise
################################################################################
## Individual: Internal ########################################################
using ..IsingBoltzmann: bitstrings, @default_first_arg
using LinearAlgebra: mul!
## External ####################################################################
using Random: GLOBAL_RNG
using Reexport: @reexport
using Requires: @require

function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("RBM/cuarrays.jl")
end

export RestrictedBoltzmann,
       biastype, weightstype, nodestype,
       copyweights!,
       energy, partitionfunc, pdf, input_pdf, kldiv,
       KLDivGradKernels, update!

struct RestrictedBoltzmann{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    inputsize::Int
    hiddensize::Int
    
    ## Parameters
    inputbias::V
    hiddenbias::V
    weights::M
end

function RestrictedBoltzmann(inputbias, hiddenbias, weights)
    # This function might not be intended for public use...
    T = Base.promote_eltype(inputbias, hiddenbias, weights)

    inputbias, hiddenbias = promote(inputbias, hiddenbias)
    inputbias = convert(AbstractArray{T}, inputbias)
    hiddenbias = convert(AbstractArray{T}, hiddenbias)
    weights = convert(AbstractArray{T}, weights)

    RestrictedBoltzmann{T, typeof(inputbias), typeof(weights)}(
        length(inputbias), length(hiddenbias), inputbias, hiddenbias, weights
    )
end
function RestrictedBoltzmann(
        inputsize, hiddensize;
        init=zeros, inputbias_init=nothing, hiddenbias_init=nothing, weights_init=nothing
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

    RestrictedBoltzmann(inputbias, hiddenbias, weights)
end

Base.eltype(::Type{<:RestrictedBoltzmann{T}}) where T = T

function copyweights!(rbm1::RestrictedBoltzmann, rbm2::RestrictedBoltzmann)
    copyto!(rbm1.inputbias, rbm2.inputbias)
    copyto!(rbm1.hiddenbias, rbm2.hiddenbias)
    copyto!(rbm1.weights, rbm2.weights)

    rbm1
end

biastype(::Type{<:RestrictedBoltzmann{<:Any, V}}) where V = V
biastype(rbm::RestrictedBoltzmann) = biastype(typeof(rbm))

weightstype(::Type{<:RestrictedBoltzmann{<:Any, <:Any, M}}) where M = M
weightstype(rbm::RestrictedBoltzmann) = weightstype(typeof(rbm))

nodestype(::Type{<:RestrictedBoltzmann}) = BitVector
nodestype(rbm::RestrictedBoltzmann) = nodestype(typeof(rbm))

energy(rbm, inputs, hiddens) =
    -sum(rbm.inputbias.*inputs) - sum(hiddens.*rbm.hiddenbias) - sum(hiddens.*rbm.weights*inputs)
eff_energy(rbm, inputs) = -rbm.inputbias'inputs - (
    sum(eachindex(rbm.hiddenbias)) do i
        log(one(eltype(rbm)) + exp(rbm.hiddenbias[i] + rbm.weights[i, :]'inputs))
    end
)

partitionfunc(rbm) = sum(
    exp(-energy(rbm, inputs, hiddens))
    for hiddens in bitstrings(rbm.hiddensize),
        inputs  in bitstrings(rbm.inputsize)
)

function pdf(rbm; pfunc=nothing)
    isnothing(pfunc) && (pfunc = RBM.partitionfunc(rbm))

    (inputs, hiddens) -> RBM.pdf(rbm, inputs, hiddens; pfunc=pfunc)
end
function pdf(rbm, inputs, hiddens; pfunc=nothing)
    isnothing(pfunc) && (pfunc = RBM.partitionfunc(rbm))

    exp(-energy(rbm, inputs, hiddens))/pfunc
end

function input_pdf(rbm; pfunc=nothing)
    isnothing(pfunc) && (pfunc = RBM.partitionfunc(rbm))

    inputs -> RBM.input_pdf(rbm, inputs; pfunc=pfunc)
end
function input_pdf(rbm, inputs; pfunc=nothing)
    isnothing(pfunc) && (pfunc = RBM.partitionfunc(rbm))

    exp(-eff_energy(rbm, inputs))/pfunc
end

sigmoid(x) = inv(one(x)+exp(-x))

condprob_input1(rbm, h) = sigmoid.(rbm.inputbias .+ rbm.weights'h)
condprob_input1(rbm, k, h) = sigmoid(rbm.inputbias[k] + h'rbm.weights[:, k])
condprob_hidden1(rbm, σ) = sigmoid.(rbm.hiddenbias .+ rbm.weights*σ)
condprob_hidden1(rbm, k, σ) = sigmoid(rbm.hiddenbias[k] + rbm.weights[k, :]'σ)

struct AltGibbsSampler{V, Rbm<:RestrictedBoltzmann} <: Random.Sampler{NTuple{2, BitVector}}
    rbm::Rbm
    inputs::V
    hiddens::V
    cd_num::Int

    AltGibbsSampler(rbm, inputs, hiddens; cd_num) =
        new{nodestype(rbm), typeof(rbm)}(rbm, inputs, hiddens, cd_num)
end

AltGibbsSampler(init::UndefInitializer, rbm; cd_num) =
    AltGibbsSampler(
        rbm, nodestype(rbm)(init, rbm.inputsize), nodestype(rbm)(init, rbm.hiddensize);
        cd_num=cd_num
    )
@default_first_arg(
function AltGibbsSampler(
        rng::AbstractRNG=GLOBAL_RNG, rbm::RestrictedBoltzmann, inputs0; cd_num
)
    hiddens = nodestype(rbm)(undef, rbm.hiddensize)

    for k in eachindex(hiddens)
        hiddens[k] = rand(rng) <= condprob_hidden1(rbm, k, inputs0)
    end

    AltGibbsSampler(rbm, copy(inputs0), hiddens; cd_num=cd_num)
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
        rng=GLOBAL_RNG, rbm::RestrictedBoltzmann, inputs0; cd_num, copy=true
)
    ret = AltGibbsSampler(rng, rbm, inputs0; cd_num=cd_num)
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
    for _ = 1:cd.cd_num
        for k in eachindex(cd.inputs)
            @inbounds cd.inputs[k] = rand(rng) <= condprob_input1(cd.rbm, k, cd.hiddens)
        end
        for k in eachindex(cd.hiddens)
            @inbounds cd.hiddens[k] = rand(rng) <= condprob_hidden1(cd.rbm, k, cd.inputs)
        end
    end

    copy ? (Base.copy(cd.inputs), Base.copy(cd.hiddens)) : (cd.inputs, cd.hiddens)
end

function Random.rand!(rng::AbstractRNG, (inputs, hiddens), cd::AltGibbsSampler)
    for _ = 1:cd.cd_num
        for k in eachindex(cd.inputs)
            @inbounds inputs[k] = cd.inputs[k] = rand(rng) <= condprob_input1(cd.rbm, k, cd.hiddens)
        end
        for k in eachindex(cd.hiddens)
            @inbounds hiddens[k] = cd.hiddens[k] = rand(rng) <= condprob_input1(cd.rbm, k, cd.hiddens)
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
function kldiv(rbm, target_pdf::Function; pfunc=nothing)
    isnothing(pfunc) && (pfunc = RBM.partitionfunc(rbm))

    sum(bitstrings(rbm.inputsize)) do inputs
        p = target_pdf(inputs)
        p*log(p/input_pdf(rbm, inputs; pfunc=pfunc))
    end
end
## Approximation
function kldiv(rbm, batch; pfunc=nothing)
    isnothing(pfunc) && (pfunc = RBM.partitionfunc(rbm))

    L = length(batch)

    avg_log_likelihood = mean(batch) do σ
        log(input_pdf(rbm, σ; pfunc=pfunc))
    end

    entrop = entropy(batch)

    -avg_log_likelihood - entrop
end

module KLDivGradKernels
    using ..RBM
    using ..RBM: AltGibbsSampler
    export KLDivGradKernel, ExactKernel, ApproxKernel, CuExactKernel, CuApproxKernel

    ## Subtypes must implement the method
    ##     σgrad, hgrad, Wgrad = (::KLDivGradKernel)(rng, rbm, data)
    abstract type KLDivGradKernel end

    struct Grad{V, M}
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
            T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, N<:AbstractVector{Bool}
    } <: KLDivGradKernel
        σh_sampler::AltGibbsSampler{N, RestrictedBoltzmann{T, V, M}}
        grad::Grad{V, M}
        condavg_h::V
        hσ_prod::M
    end
    ExactKernel(rbm; cd_num) =
        ExactKernel{eltype(rbm), biastype(rbm), weightstype(rbm), nodestype(rbm)}(
            AltGibbsSampler(undef, rbm; cd_num=cd_num),
            Grad(rbm),
            biastype(rbm)(undef, rbm.hiddensize),
            weightstype(rbm)(undef, rbm.hiddensize, rbm.inputsize)
        )

    struct ApproxKernel{
            T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, N<:AbstractVector{Bool}
    } <: KLDivGradKernel
        σh_sampler::AltGibbsSampler{N, RestrictedBoltzmann{T, V, M}}
        grad::Grad{V, M}
        pos_h::N
        hσ_prod::M
    end
    ApproxKernel(rbm; cd_num) =
        ApproxKernel{eltype(rbm), biastype(rbm), weightstype(rbm), nodestype(rbm)}(
            AltGibbsSampler(undef, rbm; cd_num=cd_num),
            Grad(rbm),
            nodestype(rbm)(undef, rbm.hiddensize),
            weightstype(rbm)(undef, rbm.hiddensize, rbm.inputsize)
        )
end
@reexport using .KLDivGradKernels

condavg_hidden!(out, rbm, σ) = out .= sigmoid.(rbm.hiddenbias .+ mul!(out, rbm.weights, σ))
## Exact hidden conditional mean.
function (kern::ExactKernel)(rng, rbm, minibatch)
    z = zero(eltype(rbm))
    fill!(kern.grad.inputs, z)
    fill!(kern.grad.hiddens, z)
    fill!(kern.grad.weights, z)

    for σ⁺ in minibatch
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
    L = length(minibatch)
    kern.grad.inputs  ./= L
    kern.grad.hiddens ./= L
    kern.grad.weights ./= L

    kern.grad.inputs, kern.grad.hiddens, kern.grad.weights
end

## Approximate hidden conditional mean.
function (kern::ApproxKernel)(rng, rbm, minibatch)
    z = zero(eltype(rbm))
    fill!(kern.grad.inputs, z)
    fill!(kern.grad.hiddens, z)
    fill!(kern.grad.weights, z)

    for σ⁺ in minibatch
        kern.pos_h .= altgibbs!(rng, kern.σh_sampler, σ⁺; copy=false)
        σ⁻, h⁻ = rand(rng, kern.σh_sampler; copy=false)

        kern.grad.inputs  .+= σ⁻ .- σ⁺
        kern.grad.hiddens .+= h⁻ .- kern.pos_h
        kern.grad.weights .+= mul!(kern.hσ_prod, h⁻, σ⁻')
        kern.grad.weights .-= mul!(kern.hσ_prod, kern.pos_h, σ⁺')
    end
    L = length(minibatch)
    kern.grad.inputs  ./= L
    kern.grad.hiddens ./= L
    kern.grad.weights ./= L

    kern.grad.inputs, kern.grad.hiddens, kern.grad.weights
end

@default_first_arg function update!(rng::AbstractRNG=GLOBAL_RNG, rbm, kern, opt, minibatch)
    σgrad, hgrad, Wgrad = kern(rng, rbm, minibatch)

    rbm.inputbias  .-= Optimise.apply!(opt, rbm.inputbias, σgrad)
    rbm.hiddenbias .-= Optimise.apply!(opt, rbm.hiddenbias, hgrad)
    rbm.weights    .-= Optimise.apply!(opt, rbm.weights, Wgrad)

    rbm
end

end # module RBM
