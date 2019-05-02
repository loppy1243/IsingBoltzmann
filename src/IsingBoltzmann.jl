### NOTE: The Ising model uses spins ∈ {+-1}; the Boltzmann machine uses variables ∈ {0,1}.
### These have to be transformed between properly!

module IsingBoltzmann
using Plots
using Random
using Reexport: @reexport
export bitstrings

cartesian_prod(itr, n) = Iterators.product(fill(itr, n)...)

struct BitStringIter; nbits::Int end
function Base.iterate(iter::BitStringIter)
    st_iter = cartesian_prod((false, true), iter.nbits)
    x = iterate(st_iter); isnothing(x) && return nothing
    val, st_st = x

    buf = BitVector(undef, iter.nbits)
    buf .= val

    (copy(buf), (st_iter, st_st, buf))
end
function Base.iterate(iter::BitStringIter, (st_iter, st_st, buf))
    x = iterate(st_iter, st_st); isnothing(x) && return nothing
    val, st_st = x
    buf .= val
    (copy(buf), (st_iter, st_st, buf))
end
bitstrings(n) = BitStringIter(n)
Base.eltype(::Type{BitStringIter}) = BitVector
Base.length(iter::BitStringIter) = 2^iter.nbits

include("Sampling.jl");       @reexport using .Sampling
include("PeriodicArrays.jl")
include("Spins.jl");          @reexport using .Spins
include("Ising.jl");          @reexport using .Ising
include("RBM.jl");            @reexport using .RBM

function main1D()
    D = 1; N = 6
    nepochs = 1000
    sample_epochs = (10, 500, 1000)

#    Random.seed!(3802133156971901247)

    prob_exact = Vector{Float64}(undef, 2^N)
    prob_rbm = Dict(epoch => Vector{Float64}(undef, 2^N) for epoch in sample_epochs)
    kldivs = Vector{Float64}(undef, nepochs)

    init(dims...) = sqrt(inv(2+N)).*2.0.*(rand(dims...) .- 0.5)

    m = MetropolisIsing(spinrand(N), 1.0, 0.4, 1)
    rbm = ReducedBoltzmann(N, 2; init=init, learning_rate=0.01, cd_num=5)

    ising_pop = Vector{SpinGrid{D}}(undef, 2^N)
    for (i, σ) in zip(eachindex(ising_pop), bitstrings(N))
        σ = SpinGrid(σ)
        ising_pop[i] = σ
        prob_exact[i] = Ising.pdf(m, σ)
    end
    batchsize = 8; nbatches = 8
    ising_pop = [[ising_pop[i] for i = k*batchsize+1:(k+1)*batchsize] for k = 0:nbatches-1]

    for epoch = 1:nepochs
        kld = kldiv(rbm, x -> Ising.pdf(m, SpinGrid(x)))
        kldivs[epoch] = kld
        println("KL Div = ", kld)

        train!(rbm, ising_pop)
#        update!(rbm, x -> Ising.pdf(m, SpinGrid(x)))

        if epoch in sample_epochs
            k = 1
            for minibatch in ising_pop, σ in minibatch
                prob_rbm[epoch][k] = RBM.input_pdf(rbm, σ)
                k += 1
            end
        end
    end
    kld = kldiv(rbm, x -> Ising.pdf(m, SpinGrid(x)))
    kldivs[nepochs] = kld
    println("KL Div = ", kld)

    Plots.gr()

    plot(1:nepochs, kldivs,
        yaxis = (:log10, (1, Inf)),
        title="Ising 1D RBM KL Divergence", xlabel="Epoch", ylabel="KL Divergence"
    )
    savefig("kldiv_1D.pdf")

    plot(1:2^N, prob_exact, label="Exact",
        yscale = :log10, markershape=:auto,
        title="Ising 1D PDF", xlabel="\$\\sigma\$", ylabel="\$P(\\sigma)\$"
    )
    for epoch in sample_epochs
        plot!(1:2^N, prob_rbm[epoch], label="RBM (epochs=$epoch)", markershape=:auto)
    end
    savefig("pdf_1D.pdf")

    rbm
end

end # module IsingBoltzmann
