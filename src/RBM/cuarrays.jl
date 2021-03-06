using .CuArrays
import CuArrays.CURAND, CUDAnative

const CuRestrictedBoltzmann{T} = RestrictedBoltzmann{T, CuVector{T}, CuMatrix{T}}
const CuAltGibbsSampler{T} = AltGibbsSampler{CuVector{Bool}, CuRestrictedBoltzmann{T}}

nodestype(::Type{<:CuRestrictedBoltzmann}) = CuVector{Bool}

AltGibbsSampler(::Any, ::CuRestrictedBoltzmann, ::Any) =
    error("Must use CUDA RNG with CuArrays")
AltGibbsSampler(::CuRestrictedBoltzmann, ::Any) =
    error("Must pass CUDA RNG explicitly with CuArrays")

nodestype(::Type{<:CuRestrictedBoltzmann}) = CuVector{Bool}

cusigmoid(x) = inv(one(x) + CUDAnative.exp(-x))
condprob_input1(rbm::CuRestrictedBoltzmann, h) = cusigmoid.(rbm.inputbias .+ rbm.weights'h)
condprob_hidden1(rbm::CuRestrictedBoltzmann, σ) = cusigmoid.(rbm.hiddenbias .+ rbm.weights*σ)

function AltGibbsSampler(rng::CURAND.RNG, rbm::CuRestrictedBoltzmann, inputs0)
    hiddens = rand(rng, rbm.hiddensize) .<= condprob_hidden1(rbm, inputs0)
    AltGibbsSampler(rbm, copy(inputs0), hiddens)
end

function AltGibbsSampler!(rng::CURAND.RNG, ag::CuAltGibbsSampler, inputs0)
    copyto!(ag.inputs, inputs0)
    ag.hiddens .= rand(rng, ag.rbm.hiddensize) .<= condprob_hidden1(ag.rbm, ag.inputs)
    ag
end

function Random.rand(rng::CURAND.RNG, cd::CuAltGibbsSampler; copy=true)
    for _ = 1:cd.cd_num
        cd.inputs .= rand(rng, cd.rbm.inputsize) .<= condprob_input1(cd.rbm, cd.hiddens)
        cd.hiddens .= rand(rng, cd.rbm.hiddensize) .<= condprob_hidden1(cd.rbm, cd.inputs)
    end

    copy ? (Base.copy(cd.inputs), Base.copy(cd.hiddens)) : (cd.inputs, cd.hiddens)
end

function Random.rand!(rng::CURAND.RNG, (inputs, hiddens), cd::CuAltGibbsSampler)
    for _ = 1:cd.cd_num
        inputs  .= cd.inputs  .= rand(rng, cd.rbm.inputsize) #=
                                 =# .<= condprob_input1(cd.rbm, cd.hiddens)
        hiddens .= cd.hiddens .= rand(rng, cd.rbm.hiddensize) #=
                                 =# .<= condprob_hidden1(cd.rbm, cd.inputs)
    end

    (inputs, hiddens)
end

@eval KLDivGradKernels begin
    using CuArrays: CuVector, CuMatrix
    using ..RBM: CuRestrictedBoltzmann

    const CuExactKernel{T} = ExactKernel{T, CuVector{T}, CuMatrix{T}}
    const CuApproxKernel{T} = ApproxKernel{T, CuVector{T}, CuMatrix{T}, CuVector{Bool}}
    CuExactKernel(rbm::CuRestrictedBoltzmann; kwargs...) = ExactKernel(rbm; kwargs...)
    CuApproxKernel(rbm::CuRestrictedBoltzmann; kwargs...) = ApproxKernel(rbm; kwargs...)

    export CuExactKernel, CuApproxKernel
end
@reexport using .KLDivGradKernels
