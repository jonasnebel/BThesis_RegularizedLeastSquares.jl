using RegularizedLeastSquares
using JLArrays
using CUDA
using Plots
using Statistics
using GPUArrays

print("\n-----------------------------------\n")

iterations = 500
columns = 2
gpu = CuArray;


function single_test()
    A = rand(16, 16)
    x = rand(16, columns)
    b = A*x;
    solver = createLinearSolver(Kaczmarz, A; iterations=iterations);
    #cb = CompareSolutionCallback(deepcopy(solve!(solver, b)));
    #x_approx = solve!(solver, b, callbacks = cb)
    x_approx = solve!(solver, b)
    # display(cb.results)
    #display(x_approx)
    isapprox(x, x_approx; atol=0.1)
end

function run_function_n_times(func, n)
    true_count = 0
    for _ in 1:n
        result = func()
        if result
            true_count += 1
        end
    end
    
    true_amount = true_count
    false_amount = n - true_count
    true_percentage = true_amount/n * 100
    false_percentage = false_amount/n * 100
    print("Results with Column amount: ")
    print(columns)
    print(" and iteration amount: ")
    print(iterations)
    print("\ntotal true: ")
    display(true_amount)
    print("total false: ")
    display(false_amount)
    print("total true percentage: ")
    display(true_percentage)
    print("total false percentage: ")
    display(false_percentage)
    print("\n")
end

@time run_function_n_times(single_test, 1)
ax = 2000
ay = 2000
xcols = 1000

function dot_with_matrix_row_GPU(state_τl::CuArray{}, A::CuArray{}, x::CuArray{}, k::Int64)
    state_τl .= vec(sum(x .* view(A, k, :), dims = 1))
    #state_τl .= vec(sum(collect(Broadcast.instantiate(Broadcast.broadcasted(*, view(A, k, :), x))), dims = 1))
  end
#print("GPU: ")
#@time dot_with_matrix_row_GPU(CUDA.zeros(xcols), CUDA.rand(ax, ay), CUDA.rand(ay, xcols), 3)
#@time dot_with_matrix_row_GPU(CUDA.zeros(xcols), CUDA.rand(ax, ay), CUDA.rand(ay, xcols), 3)

function dot_with_matrix_row_cpu(state_τl::AbstractArray{T}, A::DenseMatrix{T}, x::Matrix{T}, k::Int64) where {T<:Real}
    state_τl .= vec(sum(x .* view(A, k, :), dims = 1))
    #state_τl .= vec(sum(collect(Broadcast.instantiate(Broadcast.broadcasted(*, view(A, k, :), x))), dims = 1))
  end
#print("CPU: ")
#@time dot_with_matrix_row_cpu(zeros(xcols), rand(ax, ay), rand(ay, xcols), 3)
#@time dot_with_matrix_row_cpu(zeros(xcols), rand(ax, ay), rand(ay, xcols), 3)




function analyze_and_plot_runtime(N::Int=1000)
    # Initialize arrays to store parameters and runtimes
    parameters = 1:N
    runtimes = zeros(N)
    # Run the function N times with increasing parameter
    for i in 1:N
        runtimes[i] = mean([@elapsed dot_with_matrix_row_GPU(CUDA.zeros(N), CUDA.rand(ax, ay), CUDA.rand(ay, N), 3) for _ in 1:5])
    end

    # Create the plot
    p = plot(parameters, runtimes, 
             xlabel="Parameter Value", 
             ylabel="Runtime (seconds)", 
             title="Function Runtime vs Parameter Value",
             marker=:circle,
             legend=false)
    # Display the plot
    display(p)
    savefig(p, "runtime_plot_gpu.png")
    # Return the plot object and the data
    return p, parameters, runtimes
end
#analyze_and_plot_runtime()

function analyze_and_plot_runtime(N::Int=1000)
    # Initialize arrays to store parameters and runtimes
    parameters = 1:N
    runtimes = zeros(N)
    # Run the function N times with increasing parameter
    for i in 1:N
        runtimes[i] = mean([@elapsed dot_with_matrix_row_cpu(zeros(N), rand(ax, ay), rand(ay, N), 3) for _ in 1:5])
    end

    # Create the plot
    p = plot(parameters, runtimes, 
             xlabel="Parameter Value", 
             ylabel="Runtime (seconds)", 
             title="Function Runtime vs Parameter Value",
             marker=:circle,
             legend=false)
    # Display the plot
    display(p)
    savefig(p, "runtime_plot_cpu.png")
    # Return the plot object and the data
    return p, parameters, runtimes
end
#analyze_and_plot_runtime()