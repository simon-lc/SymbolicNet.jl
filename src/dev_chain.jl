using SymbolicNet
using Symbolics
using Graphs
using StaticArrays
using BenchmarkTools
using SymbolicUtils


function chain_code_generation(fs::SVector{Nf,Function}, n_input::Int) where Nf
    # chain structure each function transform a vector input into a vector output
    # fs = [f0, f1, ... fn]
    # fn( ... (f1(f0(inputs))))
    fs_expr = Vector{Expr}()
    fs_eval = Vector{Function}()
    for i in eachindex(fs)
        # generate variables
        # @variables input_arr[1:n_input]
        # input = vectorize(input_arr)
        input = Symbolics.variables(Symbol(:x, i), 1:n_input)
        output = fs[i](input)
        # output_jacobian = Symbolics.jacobian(output, input[1:n_input])
        # output_hessian = Symbolics.hessian(output[1], input[1:n_input])
        # code generation
        push!(fs_expr, build_function(output, input)[2])
        push!(fs_eval, eval(build_function(output, input)[2]))
        # fi_jaco = eval(build_function(output_jacobian, input))
        # fi_hess = eval(build_function(output_hessian, input))
        # update
        n_output = length(output)
        n_input = n_output
    end

    return SVector{Nf,Function}(fs_eval), SVector{Nf,Expr}(fs_expr)
end


# Demonstration
function f1(x0)
    # x1 = sin.(2 .* x0) .+ x0 .+ 1.0
    x1 = x0.^2
    return x1
end

function f2(x1)
    x2 = x1.^2
    return x2
end

function f3(x1)
    x2 = cosh.(x1.^2)
    return x2
end

function f(x0)
    x1 = f1(x0)
    x2 = f2(x1)
    x3 = f3(x3)
    return x3
end




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
    @show st
    # st = Symbolics.LazyState()
    @show st
    @show "#########################################"
    ex = quote
        $([:($(custom_set_array_prefix(s.arr, ex, i, st)) = $(Symbolics.toexpr(ex, st)))
           for (i, ex) in enumerate(s.elems)]...)
        nothing
    end
    @show ex
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
    @show rhss[1]
    @show rhss[2]

    outs = [Symbolics.Sym{Any}(Symbol(:out, i)) for i = 0:2]
    xia = [Symbolics.Sym{Any}(Symbol(:x, i, :a)) for i = 1:3]
    xiv = [args[1], args[2], args[3]]
    body1 = postprocess_fbody(custom_set_array(
                               # parallel,
                               # dargs,
                               xia[1],
                               outputidxs,
                               rhss[1],
                               checkbounds,
                               skipzeros))
    body2 = postprocess_fbody(custom_set_array(
                               # parallel,
                               # dargs,
                               xia[2],
                               outputidxs,
                               rhss[1],
                               checkbounds,
                               skipzeros))
    @show body2
    body2 = postprocess_fbody(custom_set_array(
                               # parallel,
                               # dargs,
                               xia[2],
                               outputidxs,
                               rhss[2],
                               checkbounds,
                               skipzeros))
    @show body2
    # bodies = [postprocess_fbody(custom_set_array(
    #                            # parallel,
    #                            # dargs,
    #                            xia[i],
    #                            outputidxs,
    #                            rhss[i],
    #                            checkbounds,
    #                            skipzeros)) for i=1:Nf-1]
    body = postprocess_fbody(Symbolics.set_array(
                               parallel,
                               dargs[end],
                               outs[end],
                               outputidxs,
                               rhss[end],
                               checkbounds,
                               skipzeros))
    assign1 = [Symbolics.Assignment(Meta.parse("var\"x1a[$i]\""), 0.0) for i=1:2]
    assign2 = [Symbolics.Assignment(Meta.parse("var\"x2a[$i]\""), 0.0) for i=1:2]
    # assign1_ex = [Symbolics.toexpr(a, Symbolics.LazyState()) for a in assign1]
    # assign2_ex = [Symbolics.toexpr(a, Symbolics.LazyState()) for a in assign2]

    # body_expr = [assign1_ex..., assign2_ex..., getfield.(bodies[1:2], :ex)...,]# body.ex]
    # body_expr = [assign1_ex..., assign2_ex..., getfield.([body1,], :ex)...,]# body.ex]
    body_expr = [getfield.([body1, body2], :ex)...,]# body.ex]
    chain_body = Symbolics.LiteralExpr(
       quote
           $(body_expr...)
       end)
    # @show Symbolics.toexpr(Symbolics.LiteralExpr(
    #    quote
    #        $(body1...)
    #    end))
    # @show Symbolics.toexpr(Symbolics.LiteralExpr(
    #   quote
    #       $(body2...)
    #   end))
    # @show Symbolics.toexpr(body1)
    # ex1 = Symbolics.toexpr(Symbolics.toexpr(body1))
    # ex2 = Symbolics.toexpr(Symbolics.toexpr(body2))
    # chain_body = :($(ex1), $(ex2))
    chain_expr = Symbolics.Func([outs[end], dargs[1]...], [assign1..., assign2...], chain_body)
    # @show Symbolics.toexpr(chain_body)
    if !isnothing(wrap_code)
        chain_expr = wrap_code(chain_expr)
    end
    # @show chain_expr
    if expression == Val{true}
        @show "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
        st = Symbolics.LazyState()
        Symbolics.Code.union_rewrites!(st.rewrites, [xiv[1]..., xiv[2]...])

        @show Symbolics.toexpr(chain_expr, st)
        return Symbolics.toexpr(chain_expr, st)
    else
        return Symbolics._build_and_inject_function(expression_module, Symbolics.toexpr(chain_expr))
    end
end

xia = [Symbolics.Sym{Any}(Symbol(:x, i, :a)) for i = 1:3]
@show Symbolics.LazyState(xia[1])
@show Symbolics.LazyState(x1v)


Nf = 3
fs = SVector{Nf,Function}(f1,f2,f3)
N = 2

@variables x0a[1:N]
@variables x1a[1:N]
@variables x2a[1:N]
x0v = Symbolics.scalarize(x0a)
x1v = Symbolics.scalarize(x1a)
x2v = Symbolics.scalarize(x2a)
x1 = f1(x0v)
x2 = f2(x1v)
x3 = f3(x2v)
my_fct = build_chain_function(Symbolics.JuliaTarget(), [x1,x2,x3], [x0v,x1v,x2v])
my_fct
my_eval = eval(my_fct)
x0 = rand(N)
x1 = zeros(N)
x2 = zeros(N)
x3 = zeros(N)
my_eval(x3, x0)
@benchmark $my_eval($x3, $x0)

Symbolics.arguments
i = 2
ss = Symbol(:xx,i,[1:2])
@variables ss
Symbolics._parse_vars(:variables, Real, :(x)[1:2])
x1a_st = @variables x1a[1:2]
xia = [Symbolics.Sym{Any}(Symbol(:x, i, :a)) for i = 1:3]
@variables xxx[1:2]
typeof(xxx[1])
typeof(xxx[1:2])
st = Symbolics.LazyState()
Symbolics.Code.union_rewrites!(st.rewrites, [xxx..., x1a_st...])
st






body
body1 = body[1]
body2 = body[2]
dargs = body[3]
fct = body[4]
Symbolics.toexpr(fct.args)
dargs
Symbolics.toexpr(body2)
map(x->Symbolics.toexpr(x, Symbolics.LazyState()), fct.args)

body = Symbolics.Let(dargs, :(), false)
Symbolics.toexpr(body)


fff = eval(Symbolics.toexpr(Symbolics.Func(dargs, [], body1)))
Symbolics.toexpr(Symbolics.Func(dargs, [], body1))
fff([1,3.0])

bex1 = Symbolics.toexpr(body[1], Symbolics.LazyState())
bex2 = body[2]
Symbolics.toexpr(Symbolics.LiteralExpr(quote
        $bex1
    end))
Symbolics.toexpr(body[2])

bex1 = Symbolics.LiteralExpr(body[1])
bex1.head
bex2.head


body[2].ex
body[1]
bex1 = Symbolics.toexpr(body[1])

SymbolicUtils.



body_lit = Symbolics.LiteralExpr(
    quote
        $(body[2].ex)
    end)
Symbolics.toexpr(Symbolics.Func([], [], body_lit))

body_lit = Symbolics.LiteralExpr(
    quote
        $bex1
    end)

Symbolics.toexpr(Symbolics.Func([], [], body_lit))


a = 10
a = 10
a = 10
a = 10











f1s = eval(Symbolics._build_function(Symbolics.JuliaTarget(), x1v, x0v, expression=Val{true}))
f1s = eval(build_function(x1v, x0v, expression=Val{true})[2])

Symbolics._set_array(Symbolics.SerialForm(), x0v, x1v)
Symbolics.toexpr(x1v, Symbolics.LazyState())

x0 = rand(N)
x1 = rand(N)
@benchmark $f1s($x1, $x0)


@show build_function(x1v, x0v, parallel=Symbolics.SerialForm())[1]

ff = chain_assembly2(fs_eval)
ff = chain_assembly3(fs_eval)
o0 = rand(N)
o1 = zeros(N)
o2 = zeros(N)
o3 = zeros(N)
floc(fs_eval, o0, o1, o2, o3)
@benchmark $floc($fs_eval, $o0, $o1, $o2, $o3)

ff(o0, o1, o2, o3)
@benchmark $ff($o0, $o1, $o2, $o3)
ff(fs_eval, o0, o1, o2, o3)
@benchmark $ff($fs_eval, $o0, $o1, $o2, $o3)

fdd2

fs_ex
x0s = SVector{N}(rand(N))
f1_eval = fs_eval[1]
f2_eval = fs_eval[2]
o1 = zeros(N)
o2 = zeros(N)
fsss(fs_eval, x0s, o1, o2)
f(f1_eval, f2_eval, x0s, o1, o2)
@benchmark $fsss($fs_eval, $x0s, $o1, $o2)
@benchmark $f($f1_eval, $f2_eval, $x0s, $o1, $o2)



fieldnames(typeof(typeof.(fs_eval)[1]))
typeof.(fs_eval)[1].name.name
typeof.(fs_eval)[2]
typed_args = Meta.parse.(["f$i::F$i" for i=1:2])
types = Meta.parse.(["F$i" for i=1:2])
args = [Symbol(f,i) for i=1:2]
.*["a" for i = 1:10]

nest = Meta.parse(*(["f$i(" for i=2:-1:1]...) * "x" * ")"^2)
ex = :(my_fct(x::SVector{N,T}, $(typed_args...)) where {N,T,$(types...)} = $nest)
eval(ex)
my_fct(x0s, f1_eval, f2_eval)
@benchmark $my_fct($x0s, $f1_eval, $f2_eval)

function chain_process(fs::SVector{Nf,Function}, n_input::Int;
        save::Bool=true, overwrite::Bool=true) where Nf
    # 1. generate fast code for eval, jaco, hess

    # 2. save fast code for eval, jaco, hess

    # 3. load fast code into the right scope for eval, jaco, hess

    return nothing
end

function chain_code_generation(fs::SVector{Nf,Function}, n_input::Int) where Nf
    # chain structure each function transform a vector input into a vector output
    # fs = [f0, f1, ... fn]
    # fn( ... (f1(f0(inputs))))
    fs_eval = Vector{Function}()
    fs_ex = Vector{Expr}()
    for i in eachindex(fs)
        # generate variables
        @variables input_arr[1:n_input]
        @variables dummy_arr[1:n_input]
        input = vectorize(input_arr)
        dummy = vectorize(dummy_arr)
        fi = fs[i]
        output = fi(input)
        output_jacobian = Symbolics.jacobian(output, input[1:n_input])
        output_hessian = Symbolics.hessian(output[1], input[1:n_input])
        # code generation
        push!(fs_ex, build_function(output, input, dummy)[1])
        push!(fs_eval, eval(build_function(output, input)[1]))
        fi_jaco = eval(build_function(output_jacobian, input))
        fi_hess = eval(build_function(output_hessian, input))
        # update
        n_output = length(output)
        n_input = n_output
    end

    return SVector{Nf,Function}(fs_eval), SVector{Nf,Expr}(fs_ex)
end


using LinearAlgebra

function jacobian(u)
    U = u[1] * Array(Diagonal(ones(n)))
    U[2:end,1] = u[2:end]
    U[1,2:end] = u[2:end]
    return U
end

function inverse(u)
    n = length(u)
    α = -1/u[1]^2 * norm(u[2:end])^2
    β = 1 / (1 + α)
    S1 = zeros(n,n)
    S1[end,1:end-1] = u[end:-1:2]/u[1]
    S2 = zeros(n,n)
    S2[1:end-1,end] = u[end:-1:2]/u[1]
    P = zeros(n,n)
    for i = 1:n
        P[end-i+1,i] = 1
    end
    Vi = (I - S1) * (I - β * (S2 * (I - S1)))
    Ui = P * 1/u[1] * Vi * P
    return Ui
end

function inverse(u,x)
    n = length(u)
    α = -1/u[1]^2 * norm(u[2:end])^2
    β = 1 / (1 + α)

    x0 = x - [u[2:end]'*x[2:end]; zeros(n-1)]
    x1 = x - β * [0; u[2:end] * x0[1]]
    x2 = x1 - [u[2:end]'*x1[2:end]; zeros(n-1)]
    x = 1/u[1] * x2
    return xi
end


n = 3
x = rand(n)
u = rand(n)
U = jacobian(u)
U * inverse(u) * x - x
U * inverse(u,x) - x
