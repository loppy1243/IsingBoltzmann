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
julia> include("run/1D.jl")
julia> IsingBoltzmann1DRun.main()
```
The parameters of the RBM and the training can be tweaked using the `CONFIG` variable found in
`run/1D.jl`.

# It works now!
I used to have a blurb here about how it doesn't work, but it _does_ work now; for the sake of
posterity, that blurb can be found here: [README.broken.md](README.broken.md)

# References
\[1]: G. Torlai and R. G. Melko. "Learning Thermodynamics with Boltzmann Machines".
[arXiv:1606.02718v1](https://arxiv.org/abs/1606.02718v1) (2016).
