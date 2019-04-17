module RBM
using Random
using ..IsingBoltzmann: cartesian_prod
export ReducedBoltzmann, energy, partitionfunc, update!, train!, kldiv

const MFloat64 = Union{Nothing, Float64}
struct ReducedBoltzmann{T, Rng, Rf<:Ref{MFloat64}}
    inputsize::Int
    hiddensize::Int
    
    ## Parameters
    inputbias::Vector{T}
    hiddenbias::Vector{T}
    weights::Matrix{T}

    ## Hyperparameters
    learning_rate::Float64
    cd_num::Int

    ## Scratch space
    inputgrad::Vector{T}
    hiddengrad::Vector{T}
    weightsgrad::Matrix{T}

    cdavg_input::Vector{T}
    cdavg_hidden::Vector{T}
    cdavg_corr::Matrix{T}

    condavg_hidden::Vector{T}

    rng::Rng

    partitionfunc::Rf
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

    T = Base.promote_eltype(inputbias, hiddenbias, weights)
    inputbias  = convert(Vector{T}, inputbias)
    hiddenbias = convert(Vector{T}, hiddenbias)
    weights    = convert(Matrix{T}, weights)

    ref = Ref{MFloat64}(partitionfunc)

    ReducedBoltzmann{eltype(inputbias), typeof(rng), typeof(ref)}(
        inputsize, hiddensize,
        inputbias, hiddenbias, weights, learning_rate, cd_num,
        similar(inputbias), similar(hiddenbias), similar(weights),
        similar(inputbias), similar(hiddenbias), similar(weights),
        similar(hiddenbias),
        rng,
        ref
    )
end

Base.eltype(rbm::ReducedBoltzmann{T}) where T = T

energy(rbm, inputs, hiddens) =
    rbm.inputbias'inputs + hiddens'rbm.hiddenbias + hiddens'rbm.weights*inputs
eff_energy(rbm, inputs) = -rbm.inputbias'inputs - (
    sum(eachindex(rbm.hiddenbias)) do i
        log(one(eltype(rbm)) + exp(rbm.hiddenbias[i] + rbm.weights[i, :]'inputs))
    end
)

partitionfunc(rbm) = isnothing(rbm.partitionfunc[]) ? _partitionfunc(rbm) : rbm.partitionfunc[]
function _partitionfunc(rbm)
    inputs = BitVector(undef, rbm.inputsize)
    hiddens = BitVector(undef, rbm.hiddensize)
    sum = zero(eltype(rbm))
    for hidden_bits in cartesian_prod((false, true), rbm.hiddensize),
            input_bits in cartesian_prod((false, true), rbm.inputsize)
        inputs .= input_bits
        hiddens .= hidden_bits
        sum += exp(-energy(rbm, inputs, hiddens))
    end

    rbm.partitionfunc[] = sum

    sum
end

pdf(rbm, inputs, hidden) = exp(-energy(rbm, inputs, hidden))/partitionfunc(rbm)
input_pdf(rbm, inputs) = exp(-eff_energy(rbm, inputs))/partitionfunc(rbm)

sigmoid(x) = inv(one(x)+exp(-x))

condprob_input1_hidden(rbm, k, h::AbstractVector) =
    sigmoid(rbm.inputbias[k] + h'rbm.weights[:, k])
condprob_input1_hidden(rbm, k, h) = condprob_input1_hidden(rbm, k, vec(h))

condavg_input_hidden(rbm, h::AbstractVector) = sigmoid.(rbm.inputbias .+ rbm.weights'h)
condavg_input_hidden(rbm, h) = condprob_input1_hidden(rbm, vec(h))

condprob_hidden1_input(rbm, k, σ::AbstractVector) =
    sigmoid(rbm.hiddenbias[k] + rbm.weights[k, :]'σ)
condprob_hidden1_input(rbm, k, σ) = condprob_hidden1_input(rbm, k, vec(σ))

condavg_hidden_input!(rbm, σ::AbstractVector) =
    rbm.condavg_hidden .= sigmoid.(rbm.hiddenbias .+ rbm.weights*σ)
condavg_hidden_input!(rbm, σ) = condprob_hidden1_input(rbm, vec(σ))

struct RBMCD <: Random.Sampler{NTuple{2, BitVector}}
    rbm::ReducedBoltzmann
    hiddens::BitVector
    inputs::BitVector
end
RBMCD(rbm, inputs0) =
    RBMCD(rbm, BitVector(undef, rbm.hiddensize), inputs0)

function Random.rand(rng::AbstractRNG, cd::RBMCD; copy=true)
    for k in eachindex(cd.hiddens)
        cd.hiddens[k] = rand(rng) <= condprob_hidden1_input(cd.rbm, k, cd.inputs)
    end
    for k in eachindex(cd.inputs)
        cd.inputs[k] = rand(rng) <= condprob_input1_hidden(cd.rbm, k, cd.hiddens)
    end

    copy ? (copy(cd.hiddens), copy(cd.inputs)) : (cd.hiddens, cd.inputs)
end

## Exact
function kldiv(rbm, target_pdf::Function)
    inputs = BitVector(undef, rbm.inputsize)
    sum = zero(eltype(rbm))
    for bits in cartesian_prod((false, true), rbm.inputsize)
        inputs .= bits

        p = target_pdf(inputs)
        sum += p*log(p/input_pdf(rbm, inputs))
    end

    sum
end
## Approximation
function kldiv(rbm, batch)
    L = length(batch)

    log_sum = L\sum(batch) do σ
        log(input_pdf(rbm, σ))
    end
    entropy = L\sum(batch) do σ
        c = count(==(σ), batch)
        c*log(c/L)
    end

    -log_sum - entropy
end

function kldiv_grad!(rbm, batch)
    l_is = length(batch)

    z = zero(eltype(rbm))
    for σ in batch
        rbm.cdavg_input  .= z
        rbm.cdavg_hidden .= z
        rbm.cdavg_corr   .= z

        cd = RBMCD(rbm, σ)
        for _ = 1:rbm.cd_num
            h, σ2 = rand(rbm.rng, cd; copy=false)
            rbm.cdavg_input  .+= σ2
            rbm.cdavg_hidden .+= h
            rbm.cdavg_corr   .+= h*σ2'
        end
        rbm.inputgrad   .= rbm.cdavg_input  ./ rbm.cd_num
        rbm.hiddengrad  .= rbm.cdavg_hidden ./ rbm.cd_num
        rbm.weightsgrad .= rbm.cdavg_corr   ./ rbm.cd_num

        condavg_hidden_input!(rbm, σ)

        rbm.inputgrad   .-= σ
        rbm.hiddengrad  .-= rbm.condavg_hidden
        rbm.weightsgrad .-= rbm.condavg_hidden*σ'
    end
    rbm.inputgrad   ./= l_is
    rbm.hiddengrad  ./= l_is
    rbm.weightsgrad ./= l_is

    rbm.inputgrad, rbm.hiddengrad, rbm.weightsgrad
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

    for b in minibatches; update!(rbm, b) end

    perm
end

end # module RBM
