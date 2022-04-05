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
    x1 = 2 .* x0
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

function chain_assembly(fs_ev::SVector{Nf,Function}) where Nf
    fargs = Meta.parse.(["f_$i" for i=1:Nf])
    typed_fargs = Meta.parse.(["f$i::F$i" for i=1:Nf])

    oargs = Meta.parse.(["o$i" for i=0:Nf])
    typed_oargs = Meta.parse.(["o$i::Vector{T}" for i=0:Nf])

    types = Meta.parse.(["F$i" for i=1:Nf])

    body = Meta.parse.(["f$i(o$i, o$(i-1))" for i=1:Nf])
    body = [body..., Meta.parse("return o$(Nf)")]
    ex = :(f_local($(typed_fargs...), $(typed_oargs...)) where {T,$(types...)} = $(body...))
    eval(ex)

    # for i = 1:Nf
    #     eval(Meta.parse("f$i = fs_ev[$i]"))
    # end
    f_1 = fs_ev[1]
    f_2 = fs_ev[2]
    f_3 = fs_ev[3]

    # eval(ex)
    fargs = Meta.parse.(["fs_ev[$i]" for i=1:Nf])
    oargs = Meta.parse.(["o$i" for i=0:Nf])
    body = :(f_local($(fargs...), $(oargs...)))
    ex = :(f_assembled($(oargs...)) = $body)
    @show ex
    eval(ex)
    # f(o0, o1, o2, o3) = f_local(fs_ev[1], fs_ev[2], fs_ev[3], o0, o1, o2, o3)
    return f_assembled
end

#
# function _build_function(target::Symbolics.JuliaTarget, op, args...;
#                          conv = Symbolics.toexpr,
#                          expression = Val{true},
#                          expression_module = @__MODULE__(),
#                          checkbounds = false,
#                          states = Symbolics.LazyState(),
#                          linenumbers = true)
#     dargs = map((x) -> Symbolics.destructure_arg(x[2], !checkbounds, Symbol("ˍ₋arg$(x[1])")), enumerate([args...]))
#     # @show dargs
#     @show Symbolics.Func(dargs, [], op)
#     expr = Symbolics.toexpr(Symbolics.Func(dargs, [], op), states)
#     @show expr
#
#     if expression == Val{true}
#         expr
#     else
#         Symbolics._build_and_inject_function(expression_module, expr)
#     end
# end


function Symbolics._set_array(out, outputidxs, rhss::AbstractArray, checkbounds, skipzeros, )
    if outputidxs === nothing
        outputidxs = collect(eachindex(rhss))
    end
    # sometimes outputidxs is a Tuple
    ii = findall(i->!(rhss[i] isa AbstractArray) && !(skipzeros && _iszero(rhss[i])), eachindex(outputidxs))
    jj = findall(i->rhss[i] isa AbstractArray, eachindex(outputidxs))
    exprs = []
    setterexpr = Symbolics.SetArray(!checkbounds,
                          out,
                          [Symbolics.AtIndex(outputidxs[i],
                                   rhss[i])
                           for i in ii])
    push!(exprs, setterexpr)
    for j in jj
        push!(exprs, _set_array(LiteralExpr(:($out[$j])), nothing, rhss[j], checkbounds, skipzeros))
    end
    LiteralExpr(quote
                    $(exprs...)
                end)
end


function build_chain_function(target::Symbolics.JuliaTarget, rhss::AbstractArray, args...;
                       expression = Val{true},
                       expression_module = @__MODULE__(),
                       checkbounds = false,
                       postprocess_fbody=ex -> ex,
                       linenumbers = false,
                       outputidxs=nothing,
                       skipzeros = false,
                       wrap_code = (nothing, nothing),
                       fillzeros = skipzeros && !(rhss isa SparseMatrixCSC),
                       parallel=Symbolics.SerialForm(), kwargs...)

    dargs = map((x) -> Symbolics.destructure_arg(x[2], !checkbounds,
                                  Symbol("ˍ₋arg$(x[1])")), enumerate([args...]))
    i = findfirst(x->x isa Symbolics.DestructuredArgs, dargs)
    similarto = i === nothing ? Array : dargs[i].name
    oop_expr = Symbolics.Func(dargs, [],
                    postprocess_fbody(Symbolics.make_array(parallel, dargs, rhss, similarto)))

    if !isnothing(wrap_code[1])
        oop_expr = wrap_code[1](oop_expr)
    end

    out = Symbolics.Sym{Any}(:ˍ₋out)
    ip_expr = Symbolics.Func([out, dargs...], [],
                   postprocess_fbody(Symbolics.set_array(parallel,
                                               dargs,
                                               out,
                                               outputidxs,
                                               rhss,
                                               checkbounds,
                                               skipzeros)))

    if !isnothing(wrap_code[2])
        ip_expr = wrap_code[2](ip_expr)
    end

    if expression == Val{true}
        out0 = Symbolics.Sym{Any}(:out0)
        out1 = Symbolics.Sym{Any}(:out1)
        X1a = Symbolics.Sym{Any}(:X1a)
        body1 = postprocess_fbody(Symbolics.set_array(parallel,
                                    dargs,
                                    out0,
                                    outputidxs,
                                    rhss,
                                    checkbounds,
                                    skipzeros))
        body2 = postprocess_fbody(Symbolics.set_array(parallel,
                                    dargs,
                                    out1,
                                    outputidxs,
                                    rhss,
                                    checkbounds,
                                    skipzeros))
        # assign0 = [Symbolics.Assignment(Meta.parse("var\"x1a[$i]\""), 0.0) for i=1:3]
        assign0 = [Symbolics.Assignment(Meta.parse("X1a[$i]"), 0.0) for i=1:3]
        assign_ex = [Symbolics.toexpr(a, Symbolics.LazyState()) for a in assign0]

        body_expr = [assign_ex..., body1.ex, body2.ex]
        chain_body = Symbolics.LiteralExpr(
            quote
                $(body_expr...)
            end)
        chain_expr = Symbolics.Func([out1, out0, dargs...], [], chain_body)

        return Symbolics.toexpr(oop_expr), Symbolics.toexpr(ip_expr), Symbolics.toexpr(chain_expr), (body1, body2, dargs, chain_expr)
    else
        return Symbolics._build_and_inject_function(expression_module, Symbolics.toexpr(oop_expr)),
        Symbolics._build_and_inject_function(expression_module, Symbolics.toexpr(ip_expr))
    end
end

Nf = 3
fs = SVector{Nf,Function}(f1,f2,f3)
N = 2

@variables x0a[1:N]
x0v = Symbolics.scalarize(x0a)
x1v = f1(x0v)
_, _, my_fct, body = build_chain_function(Symbolics.JuliaTarget(), x1v, x0v)
my_fct
my_eval = eval(my_fct)
x0 = rand(N)
x1 = zeros(N)
x2 = zeros(N)
my_eval(x2, x1, x0)
@benchmark my_eval($x2, $x1, $x0)
my_eval2(x2, x0, x1::Vector{T}=zeros(N)) where {N,T} = my_eval(x2, x1, x0)
my_eval2(x2, x1)



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
