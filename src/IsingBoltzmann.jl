### NOTE: The Ising model uses spins ∈ {+-1}; the Boltzmann machine uses variables ∈ {0,1}.
### These have to be transformed between properly!

module IsingBoltzmann

cartesian_prod(itr, n) = Iterators.product(fill(itr, n)...)

include("Sampling.jl");       using .Sampling
include("PeriodicArrays.jl")
include("Spins.jl");          using .Spins
include("Ising.jl");          using .Ising
include("RBM.jl");            using .RBM

function main1D()
    D = 1; N = 6
    m = MetropolisIsing(spinrand(N), 1.0, 1.0, 1)
    rbm = ReducedBoltzmann(N, N; init=rand, learning_rate=0.01, cd_num=5)

    ising_pop = Vector{SpinGrid{D}}(undef, 2^N)
    for (i, bits) in zip(eachindex(ising_pop), cartesian_prod((SPINDN, SPINUP), N))
        spins = SpinGrid{D}(undef, N)
        spins .= bits
        ising_pop[i] = spins
    end
    batchsize = 8; nbatches = 8
    ising_pop = [[ising_pop[i] for i = k*batchsize+1:(k+1)*batchsize] for k = 0:nbatches-1]

    for epoch = 1:1000
        println(kldiv(rbm, x -> Ising.pdf(m, SpinGrid(x))))
        train!(rbm, ising_pop)
    end
    
    rbm
end

end # module IsingBoltzmann
