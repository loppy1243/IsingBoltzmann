macro default_first_arg(func_def)
    @assert func_def isa Expr
    @assert func_def.head === :function || func_def.head === :(=) 
                                        
    call_expr = if func_def.args[1].head === :where
        func_def.args[1].args[1]
    else
        @assert func_def.args[1].head === :call
        func_def.args[1]
    end
    idx = call_expr.args[2] isa Expr && call_expr.args[2].head === :parameters ? 3 : 2
    first_param = call_expr.args[idx]
    @assert first_param isa Expr && first_param.head === :kw
    call_expr.args[idx] = first_param.args[1]

    func_def_default = deepcopy(func_def)
    call_expr = if func_def.args[1].head === :where
        func_def_default.args[1].args[1]
    else
        func_def_default.args[1]
    end
    deleteat!(call_expr.args, idx)

    name(s::Symbol) = s
    name(s::Expr) = name(s.args[1])
    func_def_default.args[2] = quote
        $(name(first_param)) = $(first_param.args[2])
        $(func_def_default.args[2].args...)
    end

    quote
        $(esc(func_def))
        $(esc(func_def_default))
    end
end

cartesian_pow(itr, n) = Iterators.product(ntuple(_ -> itr, n)...)

struct BitStringIter; nbits::Int end
function Base.iterate(iter::BitStringIter)
    st_iter = cartesian_pow((false, true), iter.nbits)
    x = iterate(st_iter); isnothing(x) && return nothing
    val, st_st = x

    buf = BitVector(undef, iter.nbits)
    buf .= val

    (copy(buf), (st_iter, st_st, buf))
end
function Base.iterate(iter::BitStringIter, (st_iter, st_st, buf))
    x = iterate(st_iter, st_st); isnothing(x) && return nothing
    val, st_st = x
    buf .= val
    (copy(buf), (st_iter, st_st, buf))
end
bitstrings(n) = BitStringIter(n)
Base.eltype(::Type{BitStringIter}) = BitVector
Base.length(iter::BitStringIter) = 2^iter.nbits

function batch(x, batchsize)
    L = size(x, ndims(x))
    nbatches = div(L - 1, batchsize) + 1
    [[x[i] for i = k*batchsize+1:min(L, (k+1)*batchsize)] for k=0:nbatches-1]
end
