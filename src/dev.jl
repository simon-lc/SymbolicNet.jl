using SymbolicNet
using Symbolics
using Graphs
using StaticArrays
using BenchmarkTools

function vectorize(a::Symbolics.Arr{Num, 1})
    v = [a[i] for i=1:length(a)]
    return v
end

# @variables a b c1 c2 c3 d e g
#
# # Multiple argument matrix
# h = [a + b + c1 + c2,
#      c3 + d + e + g,
#      0] # uses the same number of arguments as our application
# h_julia(a, b, c, d, e, g) = [a[1] + b[1] + c[1] + c[2],
#                              c[3] + d[1] + e[1] + g[1],
#                              0]
# function h_julia!(out, a, b, c, d, e, g)
#     out .= [a[1] + b[1] + c[1] + c[2], c[3] + d[1] + e[1] + g[1], 0]
# end
#
# h_str = Symbolics.build_function(h, [g])

function f0(x0)
    x1 = 2 .* x0
    return x1
end

function f1(x1)
    x2 = x1.^2
    return x2
end

function f(x0)
    x1 = f0(x0)
    x2 = f1(x1)
    return x2
end


x0 = [1.0]
x2 = f(x0)
@variables x0v_[1:30]
x0v = vectorize(x0v_)
x1v = f0(x0v)
x2v = f1(x1v)
build_function(x1v, x0v)

x2v_grad = Symbolics.jacobian(x2v, x0v)
x2v_hess = Symbolics.hessian(x2v[1], x1v[1:2])


f0s = eval(build_function(x1v, x0v)[1])
f1s = eval(build_function(x2v, x1v)[1])
f01s = eval(build_function(x2v, x0v)[1])

x0s = SVector{1}(1.0)
@benchmark x1s = $f0s($x0s)
@benchmark x2s = $f1s($x1s)
@benchmark x2s_ = $f01s($x0s)



function vector_chain(fs::SVector{Nf,Function}, n_input::Int) where Nf
    # chain structure each function transform a vector input into a vector output
    # fs = [f0, f1, ... fn]
    # fn( ... (f1(f0(inputs))))
    fs_eval = Vector{Function}()
    fs_ex = Vector{Expr}()
    for i in eachindex(fs)
        # generate variables
        @variables input_arr[1:n_input]
        input = vectorize(input_arr)
        fi = fs[i]
        output = fi(input)
        output_jacobian = Symbolics.jacobian(output, input[1:n_input])
        output_hessian = Symbolics.hessian(output[1], input[1:n_input])
        # code generation
        push!(fs_ex, build_function(output, input)[1])
        push!(fs_eval, eval(build_function(output, input)[1]))
        fi_jaco = eval(build_function(output_jacobian, input))
        fi_hess = eval(build_function(output_hessian, input))
        # update
        n_output = length(output)
        n_input = n_output
    end

    f1 = fs_eval[1]
    f2 = fs_eval[2]
    function f_eval(input)
        output = f1(input)
        output = f2(output)
        return output
    end
    return SVector{Nf,Function}(fs_eval), SVector{Nf,Expr}(fs_ex), f_eval
end

function eval_chain(fs::SVector{Nf,Function}) where Nf
    # f_exprs = [Meta.parse("f$i = fs[$i] ") for i=1:Nf]
    # f_exprs = [f_exprs; Meta.parse("f1")]
    # expr = quote
    #     function f(fs)
    #         $f_exprs[1]
    #         $f_exprs[2]
    #         return $f_exprs[3]
    #     end
    # end
    # xvar = :x
    # ex = :(x)
    # return :(fs[1]($ex))
    # return :(xvar -> xvar)
    # :(f2 = fs[2])
    # function f(input)
    #     output = f1(input)
    #     output = f2(output)
    #     return output
    # end
    return expr
end

function alloc_free(input::SVector{Ni,T}, f1, f2) where {Ni,T}
    output = f1(input)
    output = f2(output)
    return output
end
function alloc_final(input::SVector{Ni,T}, fs_eval::SVector{Nf,Function}) where {Ni,Nf,T}
    alloc_free(input, fs_eval[1], fs_eval[2])
end

fs_eval, fs_ex, _ = vector_chain(fs, 30)

typed_args = Meta.parse.(["x::Int", "y::Int", "z::Int"])
args = [:x, :y, :z]
ex = :(my_sum(a, $(typed_args...)) = sum([$(args...)]))
eval(ex)
my_sum(1,2,3,4)
my_sum(1,2,3,4.0)
fieldnames(typeof(typeof.(fs_eval)[1].name))

function f(f1::var"#497#498", f2::var"#499#500", x::SVector{N,T}) where {N,T}
    output = f1(x)
    output = f2(output)
    return output
end
function f(f1::F1, f2::F2, x::SVector{N,T}) where {N,T,F1,F2}
    output = f1(x)
    output = f2(output)
    return output
end

f1_eval = fs_eval[1]
f2_eval = fs_eval[2]
f(f1_eval, f2_eval, x0s)
@benchmark $f($f1_eval, $f2_eval, $x0s)

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

f1_eval = fs_eval[1]
f2_eval = fs_eval[2]
alloc_free(x0s, f1_eval, f2_eval)
@code_warntype alloc_free(x0s, f1_eval, f2_eval)
@benchmark $alloc_free($x0s, $f1_eval, $f2_eval)
alloc_final(x0s, fs_eval)
@code_warntype alloc_final(x0s, fs_eval)
@benchmark $alloc_final($x0s, fs_eval)


ggg(d) = d
fs_eval, fs_ex, _ = vector_chain(fs, 3)
fs_ex[1]
typeof(fs_ex[2].args[1])

fs_ex[1]

fex1 = fs_ex[1]
ex0 = Meta.parse("
    function eval_chain(x);
        fee1 = $(fs_ex[1]);
        fee2 = $(fs_ex[2]);
        fee3(x) = fee1(fee2(x));
        return fee3; end")
ex = quote
    f1 = $(fs_ex[1])
end
eval(ex0)
fg = eval_chain(fs_ex)

fee1(x0s)
fee2(x0s)
fee3(x0s)
@benchmark $fee3($x0s)
@code_warntype fee3(x0s)

@variables a[1:1]
v = vectorize(a)
w = 2v
build_function(w, v, fname="fff")

fno = function (x)
    return x
end
fno("asaaa")

fs_ex[1].args
fs_ex[1].head = :call
fs_ex[1].args
fieldnames(typeof(fs_ex[1]))

expr = eval_chain(fs_eval)

exprs = [Meta.parse("f$i = x[$i]") for i = 1:2]
ex0 = :(f1(output))
ex1  = :(f2($ex0))
expr = :(function f(x, output)
    f1 = x[1]
    f2 = x[2]
    return $ex1
    end)
# expr = :(function f(x)
#     $expr0
#     end)
fgen0 = eval(expr)
fgen0(fs_eval, x0s)
@benchmark $fgen0($fs_eval, $x0s)
@code_warntype fgen0(fs_eval, x0s)

fgen0 = eval(:(x -> x))
ex = quote
    function f(x)
        return x
    end
end
fgen1 = eval(ex)
fgen0(1.0)
fgen1(1.0)
ex0 = :(1+1)
ex1 = :(1+1)

ex = :(x -> x)

x0s = SVector{3}(1,2,3.0)
fs = SVector{2}(f0, f1)
fs_eval, _ = vector_chain(fs, 30)
f_eval = eval_chain(fs_eval, x0s)
@code_warntype eval_chain(fs_eval, x0s)
@benchmark $eval_chain($fs_eval, $x0s)


@benchmark $f_eval($x0s)
x0s = 2SVector{3}(1,2,3.0)
f_eval(x0s)


function concrete_eval_chain_gen3(input::SVector{Ni,T}, f1, f2) where {Ni,T}
    output = f1(input)
    output = f2(output)
    return output
end
f1_eval = fs_eval[1]
f2_eval = fs_eval[2]
concrete_eval_chain_gen3(x0s, f1_eval, f2_eval)
@code_warntype concrete_eval_chain_gen3(x0s, f1_eval, f2_eval)
@benchmark $concrete_eval_chain_gen3($x0s, $f1_eval, $f2_eval)



fs_eval, f_eval = vector_chain(SVector{2,Function}(f0, f1), 1)
f_eval
x0s = SVector{3}(1,2,1.0)
x1s = SVector{3}(0,0,0.0)
@benchmark $f_eval($x0s)
@benchmark $f1_eval($x0s)
@benchmark $f2_eval($x0s)

prog = "1 + 1"
ex1 = Meta.parse(prog)
typeof(ex1)
ex1.head
ex1.args

ex2 = Expr(:call, :+, 1, 1)
dump(ex2)

ex3 = Meta.parse("(4 + 4) / 2")
Meta.show_sexpr(ex3)

ex = :(a+b*c+1)
typeof(ex)
dump(ex)


ex = quote
    x = 1
    y = 2
    x + y
end
typeof(ex)
