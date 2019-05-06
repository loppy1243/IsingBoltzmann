module Spins
using Random: bitrand
using ..IsingBoltzmann: bitstrings
export Spin, flipspin, spinups, spindowns, spinrand, spinstrings, SPINS, SPINUP, SPINDN

const Spin = Bool
const SPINUP = true; const SPINDN = false
const SPINS = (SPINDN, SPINUP)
const flipspin = ~

const spindowns = falses
const spinups = trues

const spinrand = bitrand
const spinstrings = bitstrings

end # module Spins
