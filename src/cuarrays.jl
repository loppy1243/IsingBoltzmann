using .CuArrays: cu

export cuarrays_rbm

function cuarrays_rbm(config)
    init(dims) =
        sqrt(inv(config.nspins+config.nhiddens)).*2.0.*(rand(config.rng, dims...) .- 0.5) #=
        =# |> cu

    RestrictedBoltzmann(config.nspins, config.nhiddens; init=init)
end
