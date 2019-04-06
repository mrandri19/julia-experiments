using LinearAlgebra: ⋅, norm

INPUT_DIMS = 5
OUTPUT_DIMS = 3
DATASET_SIZE = 100

η = 0.01

function generate_training_data()
    x = randn((INPUT_DIMS, DATASET_SIZE))
    # A = randn((OUTPUT_DIMS,INPUT_DIMS))
    A = [1 2 3 4 5; 1 2 3 4 5; 1 2 3 4 5]
    # B = randn((OUTPUT_DIMS,1))
    B = reshape([1; 2; 3],(OUTPUT_DIMS,1))

    σ(z) = 1/(1+exp(-z))

    z = A*x .+ B
    y = σ.(z)

    @assert norm(y - σ.(A*x .+ B))^2==0. "Cost with the correct parameters should be 0"
    return x,y
end

x,y = generate_training_data()

A = randn((OUTPUT_DIMS,INPUT_DIMS))
B = randn((OUTPUT_DIMS,1))
σ(z) = 1/(1+exp(-z))

function C_A(A,B,x,y)
    (σ.(A*x .+ B)-y) .* (σ.(A*x .+ B).*(1 .- σ.(A*x .+ B))) * x'
end

function C_B(A,B,x,y)
    σ.(A*x .+ B)-y
end


for i=1:1000
    global A -= η*C_A(A,B,x,y)
    global B -= η*sum(C_B(A,B,x,y),dims=2)
    println(norm(y - σ.(A*x .+ B))^2)
end
