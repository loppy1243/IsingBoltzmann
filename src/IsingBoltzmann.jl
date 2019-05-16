### NOTE: The Ising model uses spins ∈ {+-1}; the Boltzmann machine uses variables ∈ {0,1}.
### These have to be transformed between properly!

module IsingBoltzmann
using Plots
using Random
using Reexport: @reexport
using Printf: @sprintf
export bitstrings

include("utils.jl")
include("MetropolisHastings.jl"); @reexport using .MetropolisHastings
include("PeriodicArrays.jl")
include("Spins.jl");              @reexport using .Spins
include("Ising.jl");              @reexport using .Ising
include("RBM.jl");                @reexport using .RBM

end

function main1D()
    D = 1; N = 6; N_hidden = N;
    nepochs = 1000
    sample_epochs = (10, 500, 1000)
    samplesize = 10^5
    batchsize = 50

    # Put seed  \/ here if so desired.
    Random.seed!()

    m = IsingModel(Ising.FixedBoundary, N; coupling=1.0, invtemp=0.4)
    metro = MetropolisIsingSampler(m; init=spinrand)
    ising_samples = rand(metro, samplesize)
    ising_batches = batch(ising_samples, batchsize)

    init(dims...) = sqrt(inv(N+N_hidden)).*2.0.*(rand(dims...) .- 0.5)
    rbm = RestrictedBoltzmann(N, N_hidden; init=init, learning_rate=0.01, cd_num=5)

    prob_rbm = Dict(epoch => Vector{Float64}(undef, 2^N) for epoch in sample_epochs)
    kldivs_exact = Vector{Float64}(undef, nepochs+1)
    kldivs_approx = Vector{Float64}(undef, nepochs+1)

    prob_exact = Vector{Float64}(undef, 2^N)
    for (k, σ) in spinstrings(N) |> enumerate
        prob_exact[k] = Ising.pdf(m, reshape(σ, m.size))
    end

    epochfmt(epoch) = lpad(epoch, ndigits(nepochs))
    numfmt(num) = @sprintf("%.5f", num)
    deltafmt(num) = @sprintf("%+.5f", num)
    for epoch = 1:nepochs
        kld_exact = kldiv(rbm, Ising.pdf(m))
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
    kld_exact = kldiv(rbm, Ising.pdf(m))
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

    Plots.gr()

    plot(0:nepochs, [kldivs_exact kldivs_approx], label=["Exact" "Approx"],
        yscale = :log10,
        title="Ising 1D RBM KL Divergence", xlabel="Epoch", ylabel="KL Divergence"
    )
    savefig("kldiv_1D.pdf")

    first = true
    plts = []
    for epoch in sample_epochs
        plt = plot(1:2^N, prob_exact, label="Exact",
            yscale = :log10, markershape=:auto,
            title="Epochs=$epoch", xlabel="\$\\sigma\$", ylabel="\$P(\\sigma)\$",
            legend=(first && (:top))
        )
        plot!(1:2^N, prob_rbm[epoch], label="RBM", markershape=:auto)

        push!(plts, plt)
        first = false
    end
    sz = Plots.default(:size)
    plot(plts...,
         layout=(length(sample_epochs), 1), size=(sz[1], length(sample_epochs)*sz[2])
    )
    savefig("pdf_1D.pdf")

    rbm
end

end # module IsingBoltzmann
