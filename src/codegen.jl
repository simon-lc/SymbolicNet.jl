struct CustomSetArray
    inbounds::Bool
    arr
    elems  # Either iterator of Pairs or just an iterator
end

function custom_set_array_prefix(x, ex, i, st)
    expr = :($(Symbolics.toexpr(x, st))[$(ex isa Symbolics.AtIndex ? ex.i : i)])
    Symbol(expr)
end

function Symbolics.toexpr(s::CustomSetArray, st)
    ex = quote
        $([:($(custom_set_array_prefix(s.arr, ex, i, st)) = $(Symbolics.toexpr(ex, st)))
           for (i, ex) in enumerate(s.elems)]...)
        nothing
    end
    s.inbounds ? :(@inbounds $ex) : ex
end

function custom_set_array(out, outputidxs, rhss::AbstractArray, checkbounds, skipzeros, var::Bool=true)
    if outputidxs === nothing
        outputidxs = collect(eachindex(rhss))
    end
    # sometimes outputidxs is a Tuple
    ii = findall(i->!(rhss[i] isa AbstractArray) && !(skipzeros && _iszero(rhss[i])), eachindex(outputidxs))
    jj = findall(i->rhss[i] isa AbstractArray, eachindex(outputidxs))
    exprs = []
    setterexpr = CustomSetArray(!checkbounds,
                          out,
                          [Symbolics.AtIndex(outputidxs[i],
                                   rhss[i])
                           for i in ii])
    push!(exprs, setterexpr)
    for j in jj
        push!(exprs, Symbolics._set_array(LiteralExpr(:($out[$j])), nothing, rhss[j], checkbounds, skipzeros))
    end
    Symbolics.LiteralExpr(quote
                    $(exprs...)
                end)
end

function get_variable_name(x::Num)
    name = String(Symbol(x))[10:end]
    name = split(name, ",")[1]
    return Symbol(name)
end

function get_variable_name(x::Vector{Num})
    get_variable_name(x[1])
end
