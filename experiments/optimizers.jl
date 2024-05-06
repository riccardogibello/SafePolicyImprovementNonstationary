using Printf

struct AdamParams{T} <: Any where {T}
    α::T
    β1::T
    β2::T
    ϵ::T
    t::Array{Int,1}

    # Moving average of the gradients, which represents the mean direction of the gradients.
    m::Array{T}

    # Moving average of the squared gradients, which represents the variance of the gradients.
    v::Array{T}

    # Update step for the parameters of the policy.
    Δ::Array{T}

    function AdamParams(x, α::T; β1::T = 0.9, β2::T = 0.999, ϵ::T =1e-8) where {T}
        tp = eltype(x)
        m = zeros(tp, size(x))
        v = zeros(tp, size(x))
        Δ = zeros(tp, size(x))
        new{tp}(tp(α), β1, β2, ϵ, [0], m, v, Δ)
    end
end


function update!(opt::AdamParams, x, g)
    # Update the time step in the Adam parameters
    opt.t[1] += 1
    t, β1, β2, ϵ = opt.t[1], opt.β1, opt.β2, opt.ϵ

    # Update the moving average of the gradients as:
    # m_t = β1 * m_{t-1} + (1 - β1) * g
    @. opt.m *= β1
    # as t passes, the value of β1^t decreases, so the value of the gradient
    # is less affected by the previous gradients
    @. opt.m += (1.0 - β1) * g

    # Update the moving average of the squared gradients as:
    # v_t = β2 * v_{t-1} + (1 - β2) * g^2
    @. opt.v *= β2
    # as t passes, the value of β2^t decreases, so the value of the squared gradient
    # is less affected by the previous squared gradients
    @. opt.v += (1.0 - β2) * g^2

    # Recalculate the step size (learning rate) based on the initial
    # value and the computed moving averages of the gradients and squared gradients;
    # the numerator and the denominator are used to make the step size adaptive
    # to the gradients and squared gradients in a not too fast/slow manner
    α = opt.α * √(1.0 - β2^t) / (1.0 - β1^t)

    # Calculate the update step for each of the policy parameters
    # TODO what is this used for?
    @. opt.Δ = α * opt.m / (√opt.v + ϵ)

    # Apply the update step to the parameters of the policy
    # by taking into account the moving average of the gradients
    # with epsilon to avoid division by zero
    @. x += α * opt.m / (√opt.v + ϵ)
end


"""
This method optimizes the parameters given as input (i.e. "x zero") through the Adam parameters.

g is the function used to perform the off-policy natural gradient method.

x₀ is the parameters set used to model the policy.

num_iters is the number of times the optimization must be run, calculated before as a percentage.
"""
function optimize(params::AdamParams, g!, x₀, num_iters=100)
    # Initialize a vector with the same size 
    # as the theta parameter vector (describing the policy)
    G = similar(x₀)
    # Fill the vector with zeros
    fill!(G, 0.0)
    # Copy the initial theta parameter vector
    x = deepcopy(x₀)
    # For each iteration that must be performed to 
    # optimize the policy (i.e., the G vector, which is the set to x)
    for i in 1:num_iters
        print("#######################################\n")
        print("Iteration ", i, "\n")
        print("Before optimization: \n")
        print("Old policy parameters (x): ", Printf.format.(Ref(Printf.Format("%.2f")), x), "\n")
        print("New policy parameters (G): ", Printf.format.(Ref(Printf.Format("%.2f")), G), "\n")
        
        # Perform the off-policy natural gradient method 
        # to optimize the policy parameters contained in x vector
        g!(G, x)        
        
        print("After optimization: \n")
        print("Old policy parameters (x): ", Printf.format.(Ref(Printf.Format("%.2f")), x), "\n")
        print("New policy parameters (G): ", Printf.format.(Ref(Printf.Format("%.2f")), G), "\n")
        print("#######################################\n")
        # Peform the Adam optimization method to update the policy parameters
        # stored in x by using the gradient values
        update!(params, x, G)
    end

    return x
end
