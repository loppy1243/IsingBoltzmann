module IsingBoltzmann1DRun
## Internal ####################################################################
using IsingBoltzmann
## stdlib ######################################################################
import Dates
## External ####################################################################
using Plots; Plots.gr()
################################################################################
## Individual: Internal ########################################################
using IsingBoltzmann: bitstrings
## stdlib ######################################################################
using Random: MersenneTwister
using Printf: @sprintf

const CONFIG = IsingBoltzmann.AppConfig(
    spinsize=(6,),
    coupling=1.0,
    invtemp=0.4,
    boundarycond=Ising.FixedBoundary(),
    
    nhiddens=6,
    learning_rate=0.1,
    kldiv_grad_kernel=KLDivGradKernels.ExactKernel,
    kernel_kwargs=Dict(:cd_num=>5),

    nepochs=1000,
    nsamples=10^5,
    batchsize=50,
    rng=MersenneTwister(1022150136457226227)
)

function main(; sample_epochs=[10, 500, 1000], debug=false)
    isingmodel = ising(CONFIG)

    kldivs_exact = Vector{Float64}(undef, CONFIG.nepochs+1)
    kldivs_approx = Vector{Float64}(undef, CONFIG.nepochs+1)

    ising_pf = Ising.partitionfunc(isingmodel)

    debug && print("Computing Ising PDF...")
    prob_exact = Ising.pdf.(Ref(isingmodel), spinstates(isingmodel); pfunc=ising_pf)
    debug && println(" Done.")
    prob_rbm = Dict(epoch => Vector{Float64}(undef, 2^CONFIG.nspins)
        for epoch in sample_epochs
    )

    cb = callback(
        debug, ising_pf, sample_epochs, prob_exact, prob_rbm, kldivs_exact, kldivs_approx
    )
    debug && println("Creating and training RBM:")
    train(cb, isingmodel, CONFIG)

    debug && print("Creating kldiv plot...")
    plot(0:CONFIG.nepochs, [kldivs_exact kldivs_approx], label=["Exact" "Approx"],
        yscale = :log10,
        title="Ising 1D RBM KL Divergence", xlabel="Epoch", ylabel="KL Divergence"
    )
    annotate!([(0, 0, text(Dates.now(), "monospace", 12))])
    savefig("kldiv_1D.pdf")
    debug && println(" Done.")

    debug && print("Creating PDF plot...")
    first = true
    plts = []
    for epoch in sample_epochs
        plt = plot(1:2^CONFIG.nspins, prob_exact, label="Exact",
            yscale = :log10, markershape=:auto,
            title="Epochs=$epoch", xlabel="\$\\sigma\$", ylabel="\$P(\\sigma)\$",
            legend=(first && (:top))
        )
        plot!(1:2^CONFIG.nspins, prob_rbm[epoch], label="RBM", markershape=:auto)

        push!(plts, plt)
        first = false
    end
    sz = Plots.default(:size)
    plot(plts...,
         layout=(length(sample_epochs), 1),
         size=(sz[1], length(sample_epochs)*sz[2])
    )
    annotate!([(0, 0, text(Dates.now(), "monospace", 12))])
    savefig("pdf_1D.pdf")
    debug && println(" Done.")
end

callback(debug, ising_pf, sample_epochs, prob_exact, prob_rbm, kldivs_exact, kldivs_approx) =
function(epoch, nepochs, rbm, ising, ising_samples)
    epochfmt(epoch) = lpad(epoch, ndigits(nepochs))
    numfmt(num) = @sprintf("%.5f", num)
    deltafmt(num) = @sprintf("%+.5f", num)

    rbm_pf = RBM.partitionfunc(rbm)

    debug && print("    Computing kldiv_exact...")
    kld_exact = kldiv(rbm, Ising.pdf(ising; pfunc=ising_pf); pfunc=rbm_pf)
    debug && print(" Done.\n    Computing kldiv_approx...")
    kld_approx = kldiv(rbm, ising_samples; pfunc=rbm_pf)
    debug && println(" Done.")
    kldivs_exact[epoch+1] = kld_exact
    kldivs_approx[epoch+1] = kld_approx
    Δexact = kld_exact - kldivs_exact[max(1, epoch)]
    Δapprox = kld_approx - kldivs_approx[max(1, epoch)]
    println(
        epochfmt(epoch), ": ",
        "KL_approx=$(numfmt(kld_approx)), KL_exact=$(numfmt(kld_exact)), ",
        "ΔKL_approx=$(deltafmt(Δapprox)), ΔKL_exact=$(deltafmt(Δexact))"
    )

    if epoch in sample_epochs
        debug && print("    Calculating RBM PDF for epoch ", epoch, "...")
        prob_rbm[epoch] .= RBM.input_pdf.(Ref(rbm), bitstrings(rbm.inputsize); pfunc=rbm_pf)
        debug && println(" Done.")
    end
end

end # module IsingBoltzmann1DRun
