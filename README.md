Implementation in Julia of a Restricted Boltzmann Machine (RBM) on top of the Ising model
sampled with the Metropolis algorithm. (Based on the 2016 paper by Torlai and
Melko[[1]](#references).)

# Running
In a Julia prompt in this directory, run
```
julia> import Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```
This installs dependencies. To train the RBM and produce the files `kldiv_1D.pdf` and
`pdf_1D.pdf`, run
```
julia> import IsingBoltzmann
julia> IsingBoltzmann.main1D()
```
The parameters of the RBM and the training can be tweaked within that function (found in
`src/IsingBoltzmann.jl`). To produce the Ising model Metropolis Markov chain analysis plots,
run
```
julia> include("bench/metroising.jl")
julia> IsingBoltzmannBench.bench_metroising(0)
```
The parameters in this file can be tweaked as well. The function argument is how many samples
for the Markov chain to skip inbetween each actual sample. Note that increasing this increases
run time.

There are also some tests found in `test/runtests.jl`, which can be run with
```
julia> Pkg.test()
```

# It works now!
I used to have a blurb here about how it doesn't work, but it _does_ work now; for the sake of
posterity, that blurb can be found here: [README.broken.md](README.broken.md)

# References
\[1]: G. Torlai and R. G. Melko. Learning Thermodynamics with Boltzmann Machines".
[arXiv:1606.02718v1](https://arxiv.org/abs/1606.02718v1) (2016).
