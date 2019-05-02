module Spins
using ..PeriodicArrays
using Random: bitrand
export Spin, SpinGrid, flipspin, spinups, spindowns, spinrand, SPINS, SPINUP, SPINDN

const Spin = Bool
const SPINUP = true; const SPINDN = false
const SPINS = (SPINDN, SPINUP)
const flipspin = ~

const spindowns = falses
const spinups = trues

#const SpinGrid{D} = PeriodicArray{Bool, D, BitArray{D}}
#SpinGrid(a::BitArray) = SpinGrid{ndims(a)}(a)
const SpinGrid{D} = BitArray{D}
#SpinGrid(a::BitArray) = a

spinrand(dims::Dims) = SpinGrid(bitrand(dims))
spinrand(dims::Int...) = spinrand(dims)

end # module Spins
