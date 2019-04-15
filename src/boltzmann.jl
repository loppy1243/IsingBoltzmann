using Random

struct ReducedBoltzmann
    ## Parameters
    inputbias::Vector{Float64}
    hiddenbias::Vector{Float64}
    weights::Matrix{Float64}

    ## Hyperparameters
    learning_rate::Float64
    cd_num::Int

    ## Scratch space
    inputgrad::Vector{Float64}
    inputhidden::Vector{Float64}
    weightsgrad::Matrix{Float64}

    cdavg_input::Vector{Float64}
    cdavg_output::Vector{Float64}
    cdavg_corr::Vector{Float64}

    condavg_hidden::Vector{Float64}
end
Base.getproperty(rbm::ReducedBoltzmann, s::Symbol) =
    if s === :inputsize
        length(rbm.inputbias)
    elseif s === :hiddensize
        length(rbm.hiddenbias)
    else
        Base.getfield(rbm, s)

sigmoid(x) = inv(one(x)+exp(-x))

condprob_input1_hidden(rbm, k, h::AbstractVector) =
    sigmoid(rbm.inputbias[k] + h'rbm.weights[:, k])
condprob_input1_hidden(rbm, k, h) = condprob_input1_hidden(rbm, k, vec(h))

condavg_input_hidden(rbm, h::AbstractVector) = sigmoid.(rbm.inputbias .+ rbm.weights'h)
condavg_input_hidden(rbm, h) = condprob_input1_hidden(rbm, vec(h))

condprob_hidden1_input(rbm, k, σ::AbstractVector) =
    sigmoid(rbm.hiddenbias[k] + σ'rbm.weights[k, :])
condprob_hidden1_input(rbm, k, σ) = condprob_hidden1_input(rbm, k, vec(σ))

condavg_hidden_input!(rbm, σ::AbstractVector) =
    rmb.condavg_hidden .= sigmoid.(rbm.hiddenbias .+ rbm.weights*σ)
condavg_hidden_input!(rbm, σ) = condprob_hidden1_input(rbm, vec(σ))

struct RBMCD <: Random.Sampler{NTuple{2, BitVector}}
    rbm::ReducedBoltzmann
    hiddens::BitVector
    inputs::BitVector
end
RMBCD(rbm, inputs0; rng=GLOBAL_RNG) =
    RBMCD(rbm, BitVector(undef, rbm.hiddensize, inputs0))

function Random.rand(rng::AbstractRNG, cd::RBMCD; copy=true)
    for k in eachindex(cd.hiddens)
        cd.hiddens[k] = rand(rng) <= condprob_hidden1_input(cd.rbm, k, cd.inputs)
    end
    for k in eachindex(cd.inputs)
        cd.inputs[k] = rand(rng) <= condprob_input1_hidden(cd.rbm, k, cd.hiddens)
    end

    copy ? (copy(cd.hiddens), copy(cd.inputs)) : (cd.hiddens, cd.inputs)
end

function kldiv_grad!(rbm, ising_samples; rng=GLOBAL_RNG)
    l_is = length(ising_samples)

    for σ in ising_samples
        vσ = vec(σ)

        rbm.cdavg_input  .= 0.0
        rbm.cdavg_output .= 0.0
        rbm.cdavg_corr   .= 0.0

        cd = RBMCD(rbm, vσ; rng=rng)
        for _ = 1:rbm.cd_num
            h, σ2 = rand(rng, cd; copy=false)
            rbm.cdavg_input .+= σ2
            rbm.cdavg_h     .+= h
            rbm.cdavg_corr  .+= h*σ2'
        end
        rbm.inputgrad   .= rbm.cdavg_input  ./ rbm.cd_num
        rbm.hiddengrad  .= rbm.cdavg_hidden ./ rbm.cd_num
        rbm.weightsgrad .= rbm.cdavg_corr   ./ rbm.cd_num

        condavg_hidden_input!(rbm, vσ)

        rbm.inputgrad   .-= vσ
        rbm.hiddengrad  .-= rbm.condavg_h
        rbm.weightsgrad .-= rbm.condavg_h*vσ'
    end
    rbm.inputgrad   ./= l_is
    rbm.hiddengrad  ./= l_is
    rbm.weightsgrad ./= l_is

    rbm.inputgrad, rbm.hiddengrad, rbm.weightsgrad
end

function update!(rbm, ising_samples)
    kldiv_grad!(rbm, ising_samples)

    rbm.inputbias  .-= rbm.learning_rate.*rbm.inputgrad
    rbm.hiddenbias .-= rbm.learning_rate.*rbm.hiddengrad
    rbm.weights    .-= rbm.learning_rate.*rbm.weightsgrad

    nothing
end
