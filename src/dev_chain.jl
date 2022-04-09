using SymbolicNet
using Symbolics
using Graphs
using StaticArrays
using BenchmarkTools
using SymbolicUtils
using LinearAlgebra

include("codegen.jl")

function build_chain_function(target::Symbolics.JuliaTarget, rhss::AbstractArray, args;
                       expression = Val{true},
                       expression_module = @__MODULE__(),
                       checkbounds = false,
                       postprocess_fbody=ex -> ex,
                       linenumbers = false,
                       outputidxs=nothing,
                       skipzeros = false,
                       wrap_code = nothing,
                       # fillzeros = skipzeros .&& !.(rhss .isa SparseMatrixCSC),
                       fillzeros = skipzeros && !(rhss isa SparseMatrixCSC),
                       parallel=Symbolics.SerialForm(), kwargs...)

    Nf = length(rhss)
    @assert Nf == length(args)
    dargs = [map((x) -> Symbolics.destructure_arg(x[2], !checkbounds,
                                  Symbol("ˍ₋arg$(x[1])")), enumerate([args[i]])) for i=1:Nf]
    # i = [findfirst(x->x isa Symbolics.DestructuredArgs, dargs[i]) for i=1:Nf]
    # similarto = [i === nothing ? Array : dargs[j][i].name for j=1:Nf]

    out = Symbolics.Sym{Any}(:out)
    xai = [Symbolics.Sym{Any}(get_variable_name(args[i])) for i = 1:Nf]
    bodies = [postprocess_fbody(custom_set_array(
                               # parallel,
                               # dargs,
                               xai[i],
                               outputidxs,
                               rhss[i-1],
                               checkbounds,
                               skipzeros)) for i=2:Nf]
    out_body = postprocess_fbody(Symbolics.set_array(
                               parallel,
                               dargs[end],
                               out,
                               outputidxs,
                               rhss[end],
                               checkbounds,
                               skipzeros))
    body_expr = [getfield.(bodies, :ex)..., out_body.ex]
    chain_body = Symbolics.LiteralExpr(
       quote
           $(body_expr...)
       end)
    chain_expr = Symbolics.Func([out, dargs[1]...], [], chain_body)
    if !isnothing(wrap_code)
        chain_expr = wrap_code(chain_expr)
    end
    if expression == Val{true}
        st = Symbolics.LazyState()
        Symbolics.Code.union_rewrites!(st.rewrites, [args...;])
        return Symbolics.toexpr(chain_expr, st)
    else
        return Symbolics._build_and_inject_function(expression_module, Symbolics.toexpr(chain_expr))
    end
end


# Demonstration
function f1(x0)
    x1 = sin.(2 .* x0) .+ x0 .+ 1.0
    return [x1;x1] ./ norm(x0)
end

function f2(x0)
    x1 = sin.(x0 .+ 1.0).^2
    x1 = x1 .* x0
    return [x1;x1] ./ norm(x0)
end

function f3(x0)
    x1 = cosh.(x0.^2)
    return [x1;x1] ./ norm(x0)
end

function f(x0)
    x1 = f1(x0)
    x2 = f2(x1)
    x3 = f3(x2)
    return x3
end


Nf = 3
fs = SVector{Nf,Function}(f1,f2,f3)
Ni = [2,4,8,16]

@variables xa0[1:Ni[1]]
@variables xa1[1:Ni[2]]
@variables xa2[1:Ni[3]]
xv0 = Symbolics.scalarize(xa0)
xv1 = Symbolics.scalarize(xa1)
xv2 = Symbolics.scalarize(xa2)
x1 = f1(xv0)
x2 = f2(xv1)
x3 = f3(xv2)
my_fct = build_chain_function(Symbolics.JuliaTarget(), [x1,x2,x3], [xv0,xv1,xv2], checkbounds=true)
my_fct
my_eval = eval(my_fct)
x0 = rand(Ni[1])
x1 = zeros(Ni[2])
x2 = zeros(Ni[3])
x3 = zeros(Ni[4])
x3
my_eval(x3, x0)
norm(x3 - f3(f2(f1(x0))))


naive_eval = eval(build_function(f3(f2(f1(x0v))), x0v)[2])
@benchmark $my_eval($x3, $x0)
@benchmark $naive_eval($x3, $x0)

function f1s(x0::SVector{N,T}) where {N,T}
    x1 = sin.(2 .* x0) .+ x0 .+ 1.0
    return [x1;x1] ./ norm(x0)
end

function f2s(x0::SVector{N,T}) where {N,T}
    x1 = sin.(x0 .+ 1.0).^2
    x1 = x1 .* x0
    return [x1;x1] ./ norm(x0)
end

function f3s(x0::SVector{N,T}) where {N,T}
    x1 = cosh.(x0.^2)
    return [x1;x1] ./ norm(x0)
end


function fss(x0::SVector{N,T}) where {N,T}
    x1 = f1(x0)
    x2 = f2(x1)
    x3 = f3(x2)
    return x3
end


x0 = SVector{2}(1,2.0)
x1 = SVector{4}(1,1,2,3.0)
x2 = SVector{8}(1,3,4,5,4,6,2,3.0)
@code_warntype f2s(x1)
@benchmark $fss($x0)
