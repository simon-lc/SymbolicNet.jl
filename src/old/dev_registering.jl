using SymbolicNet
using Symbolics
using Graphs
using StaticArrays
using BenchmarkTools





# Demonstration
function f1(x0)
    x1 = sin.(2 .* x0) .+ x0 .+ 1.0
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
    x3 = f3(x2)
    return x3
end

x0 = Symbolics.variables(:x0, 1:2)
f1(x0)
f2(x0)
f3(x0)
f(x0)
@register_symbolic SymbolicNet.f1(x)::Vector
f(x0)
f1

x0 = Symbolics.variable(:x0, 1)[1]
x1 = Symbolics.variable(:x1, 1)[1]
# x0 = Symbolics.variables(:x0, 1:1)
# x1 = Symbolics.variables(:x1, 1:1)
foo0(a, b) = a + b
foo1(a, b) = a .+ b
@register_symbolic foo0(a, b)
@register_symbolic foo1(a, b)
Symbolics.jacobian([foo0(x0, x1) + 1.0], [x0])
foo1(x0, x1)

@register_symbolic foo(x, y::Bool) false # do not overload a duplicate promotion rule
@register_symbolic goo(x, y::Int) # `y` is not overloaded to take symbolic objects
@register_symbolic hoo(x, y)::Int # `hoo` returns `Int`


a = 10
a = 10
a = 10













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

function vectorize(a::Symbolics.Arr{Num, 1})
    v = [a[i] for i=1:length(a)]
    return v
end

# Demonstration
function f1(x0)
    x1 = sin.(2 .* x0) .+ x0 .+ 1.0
    # x1 = 2 .* x0
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


function chain_assembly2(fs_ev::SVector{Nf,Function}) where Nf
    function f_local(o0::Vector{T}, o1::Vector{T}, o2::Vector{T}, o3::Vector{T}) where T
        fs_ev[1](o1, o0)
        fs_ev[2](o2, o1)
        fs_ev[3](o3, o2)
        return o3
    end
    return f_local
end

@generated function chain_assembly3(fs_ev::SVector{Nf,Function}) where Nf
    # function f_local(o0::Vector{T}, o1::Vector{T}, o2::Vector{T}, o3::Vector{T}) where T
    #     fs_ev[1](o1, o0)
    #     fs_ev[2](o2, o1)
    #     fs_ev[3](o3, o2)
    #     return o3
    # end
    f(o0::Vector{T}, o1::Vector{T}, o2::Vector{T}, o3::Vector{T}) where T = floc(fs_ev, o0, o1, o2, o3)
    return f
end

function floc(fs_ev::SVector{Nf,Function}, o0::Vector{T}, o1::Vector{T}, o2::Vector{T}, o3::Vector{T}) where {Nf,T}
    fs_ev[1]
    fs_ev[1](o1, o0)
    # fs_ev[2](o2, o1)
    # fs_ev[3](o3, o2)
    return o3
end

Nf = 3
fs = SVector{Nf,Function}(f1,f2,f3)
N = 2
fs_eval, fs_expr = chain_code_generation(fs, N)
fs_expr

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
