using IsingBoltzmann
using Plots
using Random, Test

Plots.gr()

Random.seed!(1793908527520900801)

const ATOL = 1e-8
const ATOL_LARGE = 1e-3

_input_pdf_literal(rbm, inputs) = RBM.partitionfunc(rbm)\sum(
    exp(-energy(rbm, inputs, hiddens)) for hiddens in bitstrings(rbm.hiddensize)
)

@testset "RBM" begin
    rbm = ReducedBoltzmann(6, 6; init=rand, learning_rate=0.1, cd_num=5)
    states = bitstrings(6)

    @test all(states) do σ
        isapprox(RBM.input_pdf(rbm, σ), _input_pdf_literal(rbm, σ); atol=ATOL)
    end
end

@testset "Ising" begin
    N = 6
    m = MetropolisIsing(spinrand(N), 1.0, 0.4, 6)
    @test isapprox(
        1.0, sum(Ising.pdf(m, SpinGrid(bits)) for bits in bitstrings(N)),
        atol=ATOL
    )

    nsamples = 10^4
    samples = rand(m, nsamples)

    counts = Dict{eltype(samples), Int}()
    for σ in samples
        counts[σ] = get(counts, σ, 0) + 1
    end

#    @info "Testing rand(::MetropolisIsing)"
    @test all(keys(counts)) do σ
        p_approx = counts[σ]/nsamples
        p_exact = Ising.pdf(m, σ)
#        @info "" p_approx p_exact

        isapprox(counts[σ]/nsamples, p_exact; atol=ATOL_LARGE)
    end
end
