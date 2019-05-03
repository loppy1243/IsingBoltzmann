module IsingBoltzmannBench
using IsingBoltzmann
using Plots, StatsPlots
using Random, Statistics

kldiv(states, target::Function, data_hist) =
    kldiv(states, target, data_hist, sum(values(data_hist)))
function kldiv(target::Function, data_hist, nsamples)
    sum(keys(data_hist)) do state
        p = target(state)

        p*log(p/(data_hist[state]/nsamples))
    end
end

hist(sampler, n) = hist(Random.GLOBAL_RNG, sampler, n)
function hist(rng, sampler, n)
    counts = Dict{Random.gentype(sampler), Int}()

    for _ = 1:n
        sample = rand(rng, sampler)

        counts[sample] = get(counts, sample, 0) + 1
    end

    counts
end

SEED = 1830375382140242722
max_burn = 10^7
max_skip = 2*10^3
max_samples = 10^6

n_burn_samples = 10^5
n_autocorr_samples = 10^3
n_accept_prob_samples = 10^3

burn_sample_interval = 2*10^5
skip_sample_interval = 5
sample_log_interval = 1

N = 6

function bench_metroising(skip)
    Plots.gr()
    Plots.default(legend=false)

    m = MetropolisIsing(spinrand(N), 1.0, 0.4, 1, skip)

    exact_energy = sum(hamiltonian(m, σ)*Ising.pdf(m, σ) for σ in bitstrings(N))
    
    kldivs = Vector{Float64}(undef, div(max_burn, burn_sample_interval)+1)
    energy_rel_diffs = Vector{Float64}(undef, div(max_burn, burn_sample_interval)+1)
    rng = MersenneTwister(SEED)
    @info "Performing burn-in" max_burn burn_sample_interval n_burn_samples
    for i = 0:max_burn
        if i % burn_sample_interval == 0
            k = div(i, burn_sample_interval) + 1
            counts = hist(copy(rng), m, n_burn_samples)
            kldivs[k] = kldiv(σ -> Ising.pdf(m, σ), counts, n_burn_samples)
            energy = n_burn_samples\sum(counts[σ]*hamiltonian(m, σ) for σ in keys(counts))
            energy_rel_diffs[k] = (energy - exact_energy)/abs(exact_energy)
        end
    
        rand(rng, m)
    end
    xs = 0:burn_sample_interval:max_burn
    plot(
        xs, kldivs,
        markershape=:auto,
        xlabel="Burn-in", ylabel="KL Div (nsamples=$n_burn_samples)"
    )
    savefig("metroising_kldiv.pdf")
    plot(
        xs, energy_rel_diffs,
        markershape=:auto,
        xlabel="Burn-in", ylabel="Energy rel. err. (nsamples=$n_burn_samples)"
    )
    savefig("metroising_energy.pdf")
    
    avgmag(ss) = mean(2*s - 1 for s in ss)
    avgmag_autocorrs = Vector{Float64}(undef, div(max_skip, skip_sample_interval))
    @info "Calculating avgmag autocorrelation" max_skip skip_sample_interval n_autocorr_samples
    for (k, skip) = enumerate(1:skip_sample_interval:max_skip)
        samples = rand(copy(rng), n_autocorr_samples+skip)
    
        corr = 0.0; avg_sq = 0.0; sq_avg = 0.0
        for n = 1:n_autocorr_samples
            corr += mean(avgmag, samples[1:n])*mean(avgmag, samples[1:n+skip])
            sq_avg += mean(avgmag, samples[1:n])
            avg_sq += mean(ss -> avgmag(ss)^2, samples[1:n])
        end
        corr /= n_autocorr_samples
        sq_avg /= n_autocorr_samples
        sq_avg ^= 2
        avg_sq /= n_autocorr_samples
    
        avgmag_autocorrs[k] = (corr - sq_avg)/(avg_sq - sq_avg)
    end
    xs = 1:skip_sample_interval:max_skip
    plot(
        xs, avgmag_autocorrs,
        xlabel="Skip", ylabel="Average Magnetization Autocorrelation (nsamples=$n_autocorr_samples)"
    ), StatsPlots
    savefig("metroising_avgmag_autocorr.pdf")

    @info "Calculating acceptance probabilties" n_accept_prob_samples
    accept_probs = Vector{Float64}(undef, n_accept_prob_samples-1)
    rng2 = copy(rng)
    prev_sample = rand(rng2, m)
    for i = 1:n_accept_prob_samples-1
        cur_sample = rand(rng2, m)
        accept_probs[i] = min(1, exp(log_accept_prob(m, cur_sample, prev_sample)))
        prev_sample = cur_sample
    end
    bar(
        1:n_accept_prob_samples-1, accept_probs,
        xlabel="Sample", ylabel="Acceptance Probability"
    )
    savefig("metroising_accept_prob.pdf")
   
    @info "Creating histograms" max_samples sample_log_interval
    hists = []
    for i = 1:sample_log_interval:log10(max_samples)
        n = 10^i
        counts = hist(copy(rng), m, n)
        
        pdf_exact = [Ising.pdf(m, σ) for σ in bitstrings(N)]
        pdf_approx = [get(counts, σ, 0)/n for σ in bitstrings(N)]
        push!(
            hists,
            groupedbar(
                1:2^N, [pdf_exact pdf_approx], labels=["exact" "approx"],
                xlabel="Configuration", ylabel="Probability", title="PDF (nsamples=$n)"
            )
        )
    end
    sz = Plots.default(:size)
    plot(layout=(length(hists), 1), hists..., size=(sz[1], length(hists)*sz[2]))
    savefig("metroising_hist.pdf")

    nothing
end

end # module IsingBoltzmannBench
