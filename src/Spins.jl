module Spins
## Internal ####################################################################
using ..IsingBoltzmann: bitstrings
## stdlib ######################################################################
using Random: bitrand

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
