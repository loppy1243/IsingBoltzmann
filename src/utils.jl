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
    [view(x, k*batchsize+1:min(L, (k+1)*batchsize)) for k=0:nbatches-1]
end

"""
    @singleton [mutable ]struct Name[<:S]
        field1[::T1]=val1
        field2[::T2]=val2
        ...
    end

Creates a true singleton struct `Name` and returns the resulting object.

The singleton object of `Name` can be retieved via `Name()`, and it is always
true that `Name() === Name()`.

`Name` must _not_ be a parametric type. If the type of a field is omitted, it
defaults to the type of the corresponding value, i.e. if `T1` is omitted then
`field1::typeof(val1)`.

`Name` may be an underscore `_`, in which case the resulting `struct` name is `gensym`'d.
"""
macro singleton(struct_expr::Expr)
    struct_expr.head != :struct && error("Expected struct expression.")

    getname(s::Symbol) = s
    getname(expr::Expr) = getname(expr.args[1])

    struct_name = struct_expr.args[2]
    if struct_name isa Expr && !(struct_name.head == :(<:) && struct_name.args[1] isa Symbol)
        error("Expected non-parametric struct expression.")
    end
    if struct_name == :(_)
        struct_name = gensym("$(struct_expr.args[1] ? "mutable " : "") struct")
        struct_expr.args[2] = struct_name
    end

    struct_body = struct_expr.args[3]
    fieldnames = []
    bindings = []
    for i in eachindex(struct_body.args)
        expr = struct_body.args[i]
        if expr isa Expr
            @assert expr.head == :(=)

            fname = getname(expr)
            push!(fieldnames, fname)
            push!(bindings, expr)
            if expr.args[1] isa Symbol
                struct_body.args[i] = :($fname::typeof($fname))
            else
                struct_body.args[i] = expr.args[1]
            end
        end
    end

    inner_constuctor = :($struct_name() = new($(fieldnames...)))
    push!(struct_body.args, inner_constuctor)

    @gensym ret_sym
    esc(quote
        let $(bindings...)
            $struct_expr
            let $ret_sym = $struct_name()
                global $struct_name() = $ret_sym
                $ret_sym
            end
        end
    end)
end
