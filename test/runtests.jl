using IsingBoltzmann
using Random, Test

Plots.gr()

Random.seed!(1793908527520900801)

const ATOL = 1e-8
const RTOL = 0.5

_input_pdf_literal(rbm, inputs) = RBM.partitionfunc(rbm)\sum(
    exp(-energy(rbm, inputs, hiddens)) for hiddens in bitstrings(rbm.hiddensize)
)

@testset "RBM" begin
    rbm = RestrictedBoltzmann(6, 6; init=rand, learning_rate=0.1, cd_num=5)

    @test all(bitstrings(6)) do σ
        isapprox(RBM.input_pdf(rbm, σ), _input_pdf_literal(rbm, σ); atol=ATOL)
    end
end

@testset "Ising" begin
    m = IsingModel(Ising.FixedBoundary, 6; coupling=1.0, invtemp=0.4)
    metro = MetropolisIsingSampler(m; init=spinrand)
    @test isapprox(
        1.0, sum(Ising.pdf(m, state) for state in spinstates(m));
        atol=ATOL
    )

    nsamples = 10^5
    samples = rand(metro, nsamples)

    counts = Dict{eltype(samples), Int}()
    for σ in samples
        counts[σ] = get(counts, σ, 0) + 1
    end

    @info "Testing rand(::MetropolisIsing)"
    @test all(keys(counts)) do σ
        p_approx = counts[σ]/nsamples
        p_exact = Ising.pdf(m, σ)
        @info "" p_approx p_exact

        isapprox(counts[σ]/nsamples, p_exact; rtol=RTOL)
    end
end
