### NOTE: The Ising model uses spins ∈ {+-1}; the Boltzmann machine uses variables ∈ {0,1}.
### These have to be transformed between properly!

module IsingBoltzmann
using Plots
using Random
using Reexport: @reexport
using Printf: @sprintf
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

function batch(x, batchsize)
    L = size(x, ndims(x))
    nbatches = div(L - 1, batchsize) + 1
    [[x[i] for i = k*batchsize+1:min(L, (k+1)*batchsize)] for k=0:nbatches-1]
end

function main1D()
    D = 1; N = 6; N_hidden = N;
    nepochs = 1000
    sample_epochs = (10, 500, 1000)
    # Training with samplesize=10^5 is pretty slow, but doesn't produce better results (though
    # everything here is running serially, so it could be faster).
    samplesize = 10^3
    batchsize = 50

    # Put seed here    \/      if so desired.
    seed = Random.seed!().seed

    m = MetropolisIsing(spinrand(N), 1.0, 0.4, 1, 0)
    ising_samples = rand(m, samplesize)
    ising_batches = batch(ising_samples, batchsize)

    init(dims...) = sqrt(inv(N+N_hidden)).*2.0.*(rand(dims...) .- 0.5)
    rbm = ReducedBoltzmann(N, N_hidden; init=init, learning_rate=0.01, cd_num=5)

    prob_rbm = Dict(epoch => Vector{Float64}(undef, 2^N) for epoch in sample_epochs)
    kldivs_exact = Vector{Float64}(undef, nepochs+1)
    kldivs_approx = Vector{Float64}(undef, nepochs+1)

    prob_exact = Vector{Float64}(undef, 2^N)
    for (k, σ) in bitstrings(N) |> enumerate
        prob_exact[k] = Ising.pdf(m, SpinGrid(σ))
    end

    epochfmt(epoch) = lpad(epoch, ndigits(nepochs))
    numfmt(num) = @sprintf("%.5f", num)
    deltafmt(num) = @sprintf("%+.5f", num)
    for epoch = 1:nepochs
        kld_exact = kldiv(rbm, σ -> Ising.pdf(m, σ))
        kld_approx = kldiv(rbm, ising_samples)
        kldivs_exact[epoch] = kld_exact
        kldivs_approx[epoch] = kld_approx
        Δexact = kld_exact - kldivs_exact[max(1, epoch-1)]
        Δapprox = kld_approx - kldivs_approx[max(1, epoch-1)]
        println(
            epochfmt(epoch-1), ": ",
            "KL_approx=$(numfmt(kld_approx)), KL_exact=$(numfmt(kld_exact)), ",
            "ΔKL_approx=$(deltafmt(Δapprox)), ΔKL_exact=$(deltafmt(Δexact))"
        )

        train!(rbm, ising_batches)

        if epoch in sample_epochs
            for (k, σ) in bitstrings(N) |> enumerate
                prob_rbm[epoch][k] = RBM.input_pdf(rbm, σ)
            end
        end
    end
    kld_exact = kldiv(rbm, σ -> Ising.pdf(m, σ))
    kld_approx = kldiv(rbm, ising_samples)
    kldivs_exact[nepochs+1] = kld_exact
    kldivs_approx[nepochs+1] = kld_approx
    Δexact = kld_exact - kldivs_exact[nepochs]
    Δapprox = kld_approx - kldivs_approx[nepochs]
    println(
        epochfmt(nepochs), ": ",
        "KL_approx=$(numfmt(kld_approx)), KL_exact=$(numfmt(kld_exact)), ",
        "ΔKL_approx=$(deltafmt(Δapprox)), ΔKL_exact=$(deltafmt(Δexact))"
    )
    println("Seed = ", seed)

    Plots.gr()

    plot(0:nepochs, [kldivs_exact kldivs_approx],
        yscale = :log10,
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
