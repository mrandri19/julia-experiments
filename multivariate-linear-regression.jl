using LinearAlgebra: norm

"""
    H(x,A)
The hypothesis function
"""
function H(x,A,B)
    @assert size(x,1) == size(A,2) "x must have size (b,c) and A must have size (a,b)"
    @assert size(A,1) == size(B,1) "A must have size (a,b) and B must have size (a,b)"

    A*x .+ B
end

"""
    C(A,x,y)
The cost function
"""
function C(A,B,x,y)
    @assert size(H(x,A,B))==size(y) "Input and output have different dimentsions"
    (0.5/(size(H(x,A,B),1)*size(H(x,A,B),2))) * norm(H(x,A,B)-y)^2
end

SAMPLES=100

# The correct data
α = [1 2; 3 4; 5 6]
β = reshape([-1; -2; -3],3,1)
# Our input data
x = randn(Float64,(2,SAMPLES))
H(x,α,β)

@assert C(α,β,x,H(x,α,β))==0.0 "Cost for the correct hypotesis should be 0.0"

"""
    C_A(A,x,y)
Derivative of the cost w.r.t. A
"""
function C_A(A,B,x,y)
    (A*x .+ B - y)*(x')/size(y,1)
end

"""
    C_B(A,x,y)
Derivative of the cost w.r.t. B
"""
function C_B(A,B,x,y)
    sum((A*x .+ B - y),dims=2)/(size(x,1)*size(x,2))
end

# Learning rate
η = 0.01

# The weights we will learn
A = randn(size(α))
B = randn(size(β))

C(A,B,x,H(x,α,β))

iteration = 0
error = []
while true && iteration<100_000
    y = H(x,α,β)

    global A -= η*C_A(A,B,x,y)
    global B -= η*C_B(A,B,x,y)

    cost = C(A,B,x,y)

    if cost < eps()
        println("breaking at $(iteration)")
        break
    end
    push!(error, cost)
    global iteration += 1
end

@assert norm(A - α) < 1e-6
@assert norm(B - β) < 1e-6
