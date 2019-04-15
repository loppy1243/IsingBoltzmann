### NOTE: The Ising model uses spins ∈ {+-1}; the Boltzmann machine uses variables ∈ {0,1}.
### These have to be transformed between properly!

module IsingBoltzmann

include("Sampling.jl");       using .Sampling
include("PeriodicArrays.jl"); using .PeriodicArrays

include("spin.jl")
include("ising.jl")
include("boltzmann.jl")

end # module IsingBoltzmann
