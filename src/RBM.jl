module RBM
using Flux
using Random, Statistics
using ..IsingBoltzmann: bitstrings
export ReducedBoltzmann, energy, partitionfunc, pdf, input_pdf, update!, train!, kldiv

const MFloat64 = Union{Nothing, Float64}
struct ReducedBoltzmann{T, V, M, Rng, Rf<:Ref{MFloat64}}
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

    rng::Rng

    partitionfunc::Rf
end
function ReducedBoltzmann(inputbias, hiddenbias, weights; learning_rate, cd_num, rng=Random.GLOBAL_RNG, partitionfunc=nothing)
    ref = Ref{MFloat64}(partitionfunc)
    ReducedBoltzmann{eltype(inputbias), typeof(inputbias), typeof(weights), typeof(rng), typeof(ref)}(
        length(inputbias), length(hiddenbias),
        inputbias, hiddenbias, weights, learning_rate, cd_num,
        similar(inputbias), similar(hiddenbias), similar(weights),
        similar(hiddenbias),
        rng,
        ref
    )
end
function ReducedBoltzmann(
        inputsize, hiddensize;
        init=zeros, inputbias_init=nothing, hiddenbias_init=nothing, weights_init=nothing,
        learning_rate, cd_num,
        rng=Random.GLOBAL_RNG,
        partitionfunc=nothing
)
    inputbias  = isnothing(inputbias_init)  ? init(inputsize)             : inputbias_init(inputsize)
    hiddenbias = isnothing(hiddenbias_init) ? init(hiddensize)            : hiddenbias_init(hiddensize)
    weights    = isnothing(weights_init)    ? init(hiddensize, inputsize) : weights_init(hiddensize, inputsize)

    ref = Ref{MFloat64}(partitionfunc)

    ReducedBoltzmann{eltype(inputbias), typeof(inputbias), typeof(weights), typeof(rng), typeof(ref)}(
        inputsize, hiddensize,
        inputbias, hiddenbias, weights, learning_rate, cd_num,
        similar(inputbias), similar(hiddenbias), similar(weights),
        similar(hiddenbias),
        rng,
        ref
    )
end

Base.eltype(rbm::ReducedBoltzmann{T}) where T = T

energy(rbm, inputs, hiddens) =
    -sum(rbm.inputbias.*inputs) - sum(hiddens.*rbm.hiddenbias) - sum(hiddens.*rbm.weights*inputs)
eff_energy(rbm, inputs) = -rbm.inputbias'inputs - (
    sum(eachindex(rbm.hiddenbias)) do i
        log(one(eltype(rbm)) + exp(rbm.hiddenbias[i] + rbm.weights[i, :]'inputs))
    end
)

partitionfunc(rbm) = isnothing(rbm.partitionfunc[]) ? _partitionfunc(rbm) : rbm.partitionfunc[]
_partitionfunc(rbm) = rbm.partitionfunc[] = Flux.data(sum(
    exp(-energy(rbm, inputs, hiddens))
    for hiddens in bitstrings(rbm.hiddensize),
        inputs  in bitstrings(rbm.inputsize)
))
#partitionfunc(rbm) = sum(
#    exp(-energy(rbm, inputs, hiddens))
#    for hiddens in bitstrings(rbm.hiddensize),
#        inputs  in bitstrings(rbm.inputsize)
#)

pdf(rbm) = (inputs, hiddens) -> RBM.pdf(rbm, inputs, hiddens)
pdf(rbm, inputs, hiddens) = exp(-energy(rbm, inputs, hiddens))/partitionfunc(rbm)
input_pdf(rbm) = inputs -> RBM.input_pdf(rbm, inputs)
input_pdf(rbm, inputs) = exp(-eff_energy(rbm, inputs))/partitionfunc(rbm)

sigmoid(x) = inv(one(x)+exp(-x))

condprob_input1(rbm, k, h) = sigmoid(rbm.inputbias[k] + h'rbm.weights[:, k])
condprob_hidden1(rbm, k, σ) = sigmoid(rbm.hiddenbias[k] + rbm.weights[k, :]'σ)

#condavg_input(rbm, h) = sigmoid.(rbm.inputbias .+ rbm.weights'h)
condavg_hidden!(rbm, σ) = rbm.condavg_hidden .= sigmoid.(rbm.hiddenbias .+ rbm.weights*σ)

struct RBM_AltGibbs <: Random.Sampler{NTuple{2, BitVector}}
    rbm::ReducedBoltzmann
    inputs::BitVector
    hiddens::BitVector
end
function RBM_AltGibbs(rbm, inputs0)
    hiddens = BitVector(undef, rbm.hiddensize)

    for k in eachindex(hiddens)
        hiddens[k] = rand(rbm.rng) <= condprob_hidden1(rbm, k, inputs0)
    end

    RBM_AltGibbs(rbm, inputs0, hiddens)
end

Random.rand(cd::RBM_AltGibbs, args...; kwargs...) =
    rand(cd.rbm.rng, cd, args...; copy=copy, kwargs...)
function Random.rand(rng::AbstractRNG, cd::RBM_AltGibbs; copy=true)
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

function kldiv_grad_exact!(rbm, target_pdf)
    b = param(rbm.inputbias)
    c = param(rbm.hiddenbias)
    W = param(rbm.weights)

    rbm_tracked = ReducedBoltzmann(b, c, W; learning_rate=rbm.learning_rate, cd_num=rbm.cd_num)
    grads = Flux.gradient(() -> kldiv(rbm_tracked, target_pdf), params(b, c, W))

    rbm.inputgrad .= Flux.data(grads[b])
    rbm.hiddengrad .= Flux.data(grads[c])
    rbm.weightsgrad .= Flux.data(grads[W])

    rbm.inputgrad, rbm.hiddengrad, rbm.weightsgrad
end

function kldiv_grad!(rbm, batch)
    z = zero(eltype(rbm))
    rbm.inputgrad   .= z
    rbm.hiddengrad  .= z
    rbm.weightsgrad .= z

    L = length(batch)
    for σ in batch
        σh_sampler = RBM_AltGibbs(rbm, σ)
#        h = copy(σh_sampler.hiddens)
        for _ = 1:rbm.cd_num
            h2, σ2 = rand(σh_sampler; copy=false)
            rbm.inputgrad   .+= σ2     ./ rbm.cd_num ./ L
            rbm.hiddengrad  .+= h2     ./ rbm.cd_num ./ L
            rbm.weightsgrad .+= h2*σ2' ./ rbm.cd_num ./ L
        end

        condavg_hidden!(rbm, σ)

        rbm.inputgrad   .-= σ ./ L
        rbm.hiddengrad  .-= rbm.condavg_hidden ./ L
        rbm.weightsgrad .-= rbm.condavg_hidden*σ' ./ L
#        rbm.hiddengrad  .-= h ./ L
#        rbm.weightsgrad .-= h*σ' ./ L
    end

    rbm.inputgrad, rbm.hiddengrad, rbm.weightsgrad
end

function update!(rbm, target_pdf::Function)
    kldiv_grad_exact!(rbm, target_pdf)

    rbm.inputbias  .-= rbm.learning_rate.*rbm.inputgrad
    rbm.hiddenbias .-= rbm.learning_rate.*rbm.hiddengrad
    rbm.weights    .-= rbm.learning_rate.*rbm.weightsgrad
    rbm.partitionfunc[] = nothing

    rbm
end

function update!(rbm, batch)
    kldiv_grad!(rbm, batch)

    rbm.inputbias  .-= rbm.learning_rate.*rbm.inputgrad
    rbm.hiddenbias .-= rbm.learning_rate.*rbm.hiddengrad
    rbm.weights    .-= rbm.learning_rate.*rbm.weightsgrad
    rbm.partitionfunc[] = nothing

    rbm
end

function train!(rbm, minibatches; rng=Random.GLOBAL_RNG)
    perm = randperm(rng, size(minibatches, ndims(minibatches)))
    permute!(minibatches, perm)

    for (k, b) in minibatches |> enumerate
        update!(rbm, b)
    end

    perm
end

end # module RBM
