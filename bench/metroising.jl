module IsingBoltzmannBench
using IsingBoltzmann
using Plots, StatsPlots
using Random, Statistics
using Base.Threads: @threads

kldiv(target::Function, data_hist) =
    kldiv(target, data_hist, sum(values(data_hist)))
function kldiv(target::Function, data_hist, nsamples)
    sum(keys(data_hist)) do state
        p = target(state)

        p*log(p/(data_hist[state]/nsamples))
    end
end

randhist(x, n) = randhist(Random.GLOBAL_RNG, x, n)
randhist(rng, x, n) = randhist(rng, Random.Sampler(rng, x), n)
function randhist(rng, sampler::Random.Sampler, n)
    counts = Dict{Random.gentype(sampler), Int}()

    for _ = 1:n
        sample = rand(rng, sampler)
        counts[sample] = get(counts, sample, 0) + 1
    end

    counts
end
function hist(xs)
    counts = Dict{eltype(xs), Int}()

    for x in xs
        counts[x] = get(counts, x, 0) + 1
    end

    counts
end

histmean(h) = histmean(identity, h)
histmean(f, h) = sum(h[k]*f(k) for k in keys(h)) / sum(values(h))

SEED = 6473312523088092072

max_burn = 10^7
max_skip = 10^5
max_samples = 10^6

n_burn_samples = 10^3
n_autocorr_samples = 10^3
n_accept_prob_samples = 10^2

burn_sample_interval = 2*10^5
skip_sample_interval = 10^4
sample_log_interval = 1

N = 6

function burnin(f, name, filename, rng, metro)
    @info "Performing burn-in test on \"$name\"" name max_burn burn_sample_interval n_burn_samples
    
    vals = Vector{Float64}(undef, div(max_burn, burn_sample_interval)+1)
    for i = 0:max_burn
        if i % burn_sample_interval == 0
            k = div(i, burn_sample_interval) + 1
            counts = randhist(copy(rng), metro, n_burn_samples)
            vals[k] = f(counts)
        end
    
        rand(rng, metro)
    end
    xs = 0:burn_sample_interval:max_burn
    plot(
        xs, vals,
        markershape=:auto,
        xlabel="Burn-in", ylabel="$name (nsamples=$n_burn_samples)"
    )
    savefig(filename)

    nothing
end

function autocorr(f, name, filename, rng, metro)
    @info "Calculating $name autocorrelation" name max_skip skip_sample_interval n_autocorr_samples

    autocorrs = Vector{Float64}(undef, div(max_skip, skip_sample_interval))
    skip_range = 1:skip_sample_interval:max_skip
    samples = rand(rng, metro, n_autocorr_samples+last(skip_range))
    @threads for k = 1:length(skip_range)
        skip = skip_range[k]

        corr = 0.0; avg_sq = 0.0; sq_avg = 0.0
        for n = 1:n_autocorr_samples
            v = @view samples[1:n]
            v_skip = @view samples[1:n+skip]

            corr += mean(f, v)*mean(f, v_skip)
            sq_avg += mean(f, v)
            avg_sq += mean(ss -> f(ss)^2, v)
        end
        corr /= n_autocorr_samples
        sq_avg /= n_autocorr_samples
        sq_avg ^= 2
        avg_sq /= n_autocorr_samples
    
        autocorrs[k] = (corr - sq_avg)/(avg_sq - sq_avg)
    end
    plot(
        skip_range, autocorrs,
        markershape=:auto,
        xlabel="Skip", ylabel="Autocorrelation",
        title="$name Autocorrelation (nsamples=$n_autocorr_samples)"
    )
    savefig(filename)

    nothing
end

function accept_probs(filename, rng, metro)
    @info "Sampling acceptance probabilities" n_accept_prob_samples

    accept_probs = Vector{Float64}(undef, n_accept_prob_samples-1)
    prev_sample = currentsample(metro)
    for i = 1:n_accept_prob_samples-1
        cur_sample = Sampling.step(rng, metro)
        accept_probs[i] = min(1, exp(log_accept_prob(metro, cur_sample, prev_sample)))
        prev_sample = cur_sample
    end
    bar(
        1:n_accept_prob_samples-1, accept_probs,
        xlabel="Sample", ylabel="Acceptance Probability", title="Mean = $(mean(accept_probs))"
    )
    savefig("metroising_accept_prob.pdf")
end

function histogram_comp(exact_pdf, filename, rng, metro)
    @info "Creating comparison histograms" max_samples sample_log_interval

    sample_range = 10 .^ (1:sample_log_interval:ceil(Int, log10(max_samples)))
    hists = []
    samples = rand(rng, metro, last(sample_range))
    for n in sample_range
        counts = hist(@view samples[1:n])
        
        pdf_exact = [exact_pdf(σ) for σ in bitstrings(N)]
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
    savefig(filename)
end

function bench_metroising(skip)
    rng = MersenneTwister(SEED)

    Plots.gr()
    Plots.default(legend=false)

    m = IsingModel(Ising.FixedBoundary, N; coupling=1.0, invtemp=0.4)
    metro = MetropolisIsing(m, spinrand(rng, N); skip=skip)

#    burnin("KL Div", "metroising_kldiv.pdf", copy(rng), metro) do σ_hist
#        kldiv(Ising.pdf(m), σ_hist)
#    end
#    exact_energy = sum(hamiltonian(m, σ)*Ising.pdf(m, σ) for σ in bitstrings(N))
#    burnin("Energy", "metroising_energy.pdf", copy(rng), metro) do σ_hist
#        energy = histmean(σ -> hamiltonian(m, σ), σ_hist)
#        (energy - exact_energy)/abs(exact_energy)
#    end

    autocorr("Average Magnetization", "metroising_avgmag_autocorr.pdf", copy(rng), metro) do σ
        mean(2*s - 1 for s in σ)
    end

    accept_probs("metroising_accept_prob.pdf", copy(rng), metro)

#    histogram_comp(Ising.pdf(m), "metroising_hist.pdf", copy(rng), metro)

    nothing
end

end # module IsingBoltzmannBench
