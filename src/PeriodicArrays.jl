module PeriodicArrays
export PeriodicArray

struct PeriodicArray{T, D, A} <: AbstractArray{T, D}
    data::A
end
PeriodicArray{T, D, A}(init::UndefInitializer, dims::Dims) where {T, D, A} =
    PeriodicArray{T, D, A}(A(init, dims))
PeriodicArray{T, D, A}(init::UndefInitializer, dims::Int...) where {T, D, A} =
    PeriodicArray{T, D, A}(A(init, dims))
PeriodicArray{T, D}(data::AbstractArray{T, D}) where {T, D} =
    PeriodicArray{T, D, typeof(data)}(data)
PeriodicArray{T}(data::AbstractArray{T}) where T =
    PeriodicArray{T, ndims(data), typeof(data)}(data)
PeriodicArray{T}(init::UndefInitializer, dims::Dims) where T =
    PeriodicArray{T}(Array{T}(init, dims))
PeriodicArray{T}(init::UndefInitializer, dims::Int...) where T =
    PeriodicArray{T}(init, dims)
PeriodicArray{T}(A::Type, init::UndefInitializer, dims::Dims) where T =
    PeriodicArray{T}(A(init, dims))
PeriodicArray{T}(A::Type, init::UndefInitializer, dims::Int...) where T =
    PeriodicArray{T}(A, init, dims)
PeriodicArray(data) = PeriodicArray{eltype(data), ndims(data), typeof(data)}(data)

Base.IndexStyle(::Type{<:PeriodicArray}) = Base.IndexCartesian()
Base.size(x::PeriodicArray) = size(x.data)

modm1p1(a, b) = mod(a-1, b) + 1
Base.getindex(x::PeriodicArray{<:Any, D}, I::Vararg{Int, D}) where D =
    x.data[CartesianIndex(map(modm1p1, I, size(x.data)))]
Base.setindex!(x::PeriodicArray{<:Any, D}, v, I::Vararg{Int, D}) where D =
    x.data[CartesianIndex(map(modm1p1, I, size(x.data)))] = v

Base.similar(x::PeriodicArray, ::Type{T}, dims::Dims) where T =
    PeriodicArray(similar(x.data, T, dims))

end # module PeriodicArrays
