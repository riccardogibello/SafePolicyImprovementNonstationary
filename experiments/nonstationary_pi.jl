using Zygote
using Statistics
using Printf

include("history.jl")
include("offpolicy.jl")
include("highconfidence.jl")
include("nonstationary_modeling.jl")

"""
This method is used to calculate a non-stationary wild bootstrap confidence interval
for a given policy π. The wild bootstrapping is a resampling technique used to estimate 
the variability of an estimator when the errors are heteroscedastic 
(i.e., the variability of the errors differs across observations).
This method is called in the "HICOPI_step" method.

π: the policy object used to interact with the environment.
δ: the confidence level used to calculate the confidence interval.
tail: the tail of the confidence interval (left, right, or both).
D: the bandit history.
idxs: the array of test timesteps from the bandit history.
ϕ: the basis function that translates an integer (timestep) into a feature vector.
τ: the number of future timesteps to consider.
num_boot: the number of train samples used for bootstrapping the confidence interval.
IS: the importance sampling method used to estimate the expected return of a given policy 
    on samples which were drawn using a different policy.
rng: the random number generator used to generate random numbers.
"""
function nswildbst_CI(
    π, 
    δ, 
    tail, 
    D, 
    idxs, 
    ϕ, 
    τ, 
    num_boot, 
    IS, 
    rng
)
    L = length(D)

    # Create the feature matrices of the test indices and of the future indices
    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B, C = get_coefst(Φ, ϕτ)
    
    N = length(idxs)
    Y = zeros(Float64, N)
    estimate_return!(Y, D, idxs, π, IS)
    if tail == :left
        return wildbst_CI(
            get_preds_and_residual_t(Y, A, B, C)..., 
            B, 
            C, 
            δ, 
            num_boot, 
            rng
        )
    elseif tail == :right
        return wildbst_CI(
            get_preds_and_residual_t(Y, A, B, C)...,
            B, 
            C, 
            1-δ, 
            num_boot, 
            rng
        )
    # TODO is this case possible?
    elseif tail == :both
        return wildbst_CI(
            get_preds_and_residual_t(Y, A, B, C)..., 
            B, 
            C, 
            [δ/2.0, 1-(δ/2.0)], 
            num_boot, 
            rng
        )
    else
        println("ERROR tail: '$tail' not recognized. Returning NaN")
        return NaN
    end
    return
end

# TODO is this function even called?
function nsbst_lower_grad(θ₀, D::BanditHistory{T,TA}, idxs, π, ϕ, τ, num_boot, δ, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    L = length(D)
    N = length(idxs)
    Y = zeros(T, N)
    GY = similar(Y)
    ψ = similar(θ₀)

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B, C = get_coefst(Φ, ϕτ)

    function g!(G, θ, Y, GY, ψ, π, D, idxs, A, B, C, δ, num_boot, IS, rng)
        set_params!(π, θ)
        estimate_return!(Y, D, idxs, π, IS)
        GY .= Zygote.gradient(y->wildbst_CI(get_preds_and_residual_t(y, A, B, C)..., B, C, δ, num_boot, rng), Y)[1]
        G .= λ .* gradient_entropy(π)

        for (i,idx) in enumerate(idxs)
            gradient_logp!(ψ, π, D.actions[idx])
            @. G += ψ * Y[i] * GY[i]
        end
        G ./= length(idxs)
    end

    return (G, θ)->g!(G, θ, Y, GY, ψ, π, D, idxs, A, B, C, δ, num_boot, IS, rng)
end


# TODO is this function even called?
function nswildbs_CI(π, δ, tail, D, idxs, ϕ, τ, num_boot, aggf, IS, rng)
    L = length(D)

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B = get_coefs(Φ, ϕτ)

    N = length(idxs)
    Y = zeros(Float64, N)
    estimate_return!(Y, D, idxs, π, IS)
    if tail == :left
        return wildbs_CI(get_preds_and_residual(Y, A, B)..., B, δ, num_boot, aggf, rng)
    elseif tail == :right
        return wildbs_CI(get_preds_and_residual(Y, A, B)..., B, 1-δ, num_boot, aggf, rng)
    elseif tail == :both
        return wildbs_CI(get_preds_and_residual(Y, A, B)..., B, [δ/2.0, 1-(δ/2.0)], num_boot, aggf, rng)
    else
        println("ERROR tail: '$tail' not recognized. Returning NaN")
        return NaN
    end
end

"""
This method (called in "optimize" method) is the implementation of the natural gradient
method, a slight modification of the standard gradient method.

G: the natural gradient vector of the policy parameters to be populated.
θ: the vector containing the policy parameters.
Y: the vector containing the estimated returns of the policy π wrt the history D and the behavior policy.
ψ: a vector containing the gradients of the log probabilities of the actions.
F: the Fisher Information matrix.
π: keeps the data related to the policy used to interact with the environment.
D: the bandit history.
idxs: the array of training timesteps from the bandit history.
A: the matrix containing the W matrix.
B: the matrix containing the product between the ϕτ vector and the H matrix.
δ: the confidence interval used for the lower and upper bounds.
num_boot: the number of train samples used for bootstrapping the confidence interval.
aggf: the aggregation function used to compute the confidence interval.
λ: the entropy regularizer coefficient.
IS: the importance sampling method used to estimate the expected return of a given policy 
    on samples which were drawn using a different policy.
rng: the random number generator used to generate random numbers.
"""
function off_policy_natgrad_bs!(
    G, 
    θ, 
    Y, 
    GY, 
    ψ, 
    F, 
    π::StatelessSoftmaxPolicy, 
    D::BanditHistory{T,TA}, 
    idxs, 
    A, 
    B,
    δ, 
    num_boot, 
    aggf, 
    λ, 
    IS::TI, 
    rng
) where {T,TA,TI<:UnweightedIS}
    # Update the theta parameters of the policy by setting the current ones 
    set_params!(π, θ)
    # Compute the Fisher Information matrix, to identify how much the sample (bandit history) can explain
    # the parameters of the policy; the Fisher information matrix measures the curvature of the parameter 
    # space (areas of high curvature are dangerous, due to possibility to overshoot the minimum and diverge;
    # areas of low curvature are safe, but convergence is slow).
    compute_fisher!(F, ψ, π)
    
    # Populate the Y array with the estimated returns of the policy π wrt the history D
    # and the behavior policy
    estimate_entropyreturn!(Y, D, idxs, π, λ, IS)
    # print("Updated value of the Y vector: ", Printf.format.(Ref(Printf.Format("%.2f")), Y), "\n")
    
    # Compute the gradients of the "wildbs_CI" function, with respect to small changes 
    # in the input Y (estimated returns with the importance sampling method); 
    # GY is a vector of the same size of Y;
    GY .= Zygote.gradient(
        # Pass an anonymous function that takes a y argument (here Y) and passes it to the wildbs_CI function
        y->wildbs_CI(
            # The three dots are used to pass the results as distinct arguments;
            # A is the W matrix, while B is ϕτ * H
            get_preds_and_residual(y, A, B)..., 
            B, 
            δ, 
            num_boot, 
            aggf, 
            rng
        ),
        Y
    )[1]
    
    # Set all the parameter values for the currently computed gradient to zero
    fill!(G, 0.0)
    # For each timestep used for training
    for (i,idx) in enumerate(idxs)
        # Compute the gradient of the log probability of the action and store it in the ψ vector
        gradient_logp!(ψ, π, D.actions[idx])
        # Update the gradient vector by adding the product between: 
        # 1) the gradient of the log probability of the actions
        # 2) the estimated return of the policy π wrt the history D and the behavior policy
        # 3) the gradient of the "wildbs_CI" function
        @. G += ψ * Y[i] * GY[i]
    end

    # Normalize the gradient by the number of timesteps used for training
    G ./= length(idxs)

    # Pre-multiply the gradient with the inverse of the Fisher information matrix, 
    # which is a measure of the curvature of the parameter space. 
    # This adjusts the gradient (step size) in each direction according to the curvature, 
    # which can lead to faster and more stable learning.
    G .= inv(F) * G
end

function off_policy_natgrad_bs!(
    G, 
    θ, 
    Y, 
    GY, 
    ψ, 
    F, 
    π::StatelessNormalPolicy, 
    D::BanditHistory{T,TA}, 
    idxs, 
    A, 
    B,
    δ, 
    num_boot, 
    aggf, 
    λ, 
    IS::TI, 
    rng
) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)

    estimate_entropyreturn!(Y, D, idxs, π, λ, IS)
    GY .= Zygote.gradient(y->wildbs_CI(get_preds_and_residual(y, A, B)..., B, δ, num_boot, aggf, rng), Y)[1]

    fill!(G, 0.0)
    for (i,idx) in enumerate(idxs)
        logp = gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end
    G ./= length(idxs)
    G .= inv(F) * G
end

function off_policy_natgrad_bs_old!(
    G, 
    θ, 
    Y, 
    GY, 
    ψ, 
    F,
    π::StatelessNormalPolicy, 
    D::BanditHistory{T,TA}, 
    idxs, 
    A, 
    B, 
    δ, 
    num_boot, 
    aggf, 
    λ, 
    IS::TI, 
    rng
) where {T,TA,TI<:UnweightedIS}
    # Update the theta parameters of the policy
    set_params!(π, θ)

    compute_fisher!(F, ψ, π)

    estimate_return!(Y, D, idxs, π, IS)
    GY .= Zygote.gradient(y->wildbs_CI(get_preds_and_residual(y, A, B)..., B, δ, num_boot, aggf, rng), Y)[1]

    fill!(G, 0.0)
    G .= λ .* gradient_entropy(π)
    for (i,idx) in enumerate(idxs)
        logp = gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end

    G ./= length(idxs)
    G .= inv(F) * G
end

"""
This method returns a function that accepts two vectors of the same size of theta
and performs the off-policy natural gradient update (i.e., direction of change used
to update the policy parameters to improve the policy performance).
"""
function nsbs_lower_grad(
    θ₀, 
    D::BanditHistory{T,TA}, 
    idxs, 
    π::StatelessSoftmaxPolicy, 
    ϕ, 
    τ, 
    num_boot, 
    δ, 
    aggf,
    λ, 
    IS::TI, 
    old_ent, 
    rng
) where {T,TA,TI<:UnweightedIS}
    # Get the length of the bandit history
    L = length(D)
    # Get the number of train indexes
    N = length(idxs)
    # Create two arrays of zeros (with size N and type T) to store the 
    # rewards and their gradients
    Y = zeros(T, N)
    GY = similar(Y)
    # Create an array of the same size of the given theta parameters 
    # (i.e., the current weights that model the policy)
    ψ = similar(θ₀)

    # Create the Fourier transform of the train indexes and of the tau future indexes
    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    # Get the two matrices of weights that can be used to make predictions
    A, B = get_coefs(Φ, ϕτ)

    # Create the Fisher matrix, which is a matrix of |theta_zero| x |theta_zero|
    F = zeros(T, (length(ψ), length(ψ)))

    # If the old entropy must be used
    if old_ent
        return (G, θ)->off_policy_natgrad_bs_old!(
            G, 
            θ, 
            Y, 
            GY, 
            ψ, 
            F, 
            π, 
            D, 
            idxs, 
            A, 
            B, 
            δ, 
            num_boot, 
            aggf, 
            λ, 
            IS, 
            rng
        )
    else
        return (G, θ)->off_policy_natgrad_bs!(
            G, 
            θ, 
            Y, 
            GY, 
            ψ, 
            F, 
            π, 
            D, 
            idxs, 
            A, 
            B, 
            δ, 
            num_boot, 
            aggf, 
            λ, 
            IS, 
            rng
        )
    end
end

# TODO is this function even called?
function maximum_entropy_fit(D::BanditHistory, idxs, π::StatelessNormalPolicy)
    adist = hcat(D.actions...)
    μ = mean(adist, dims=2)[:, 1]
    σ = std(adist, dims=2)[:, 1]
    return vcat(μ, σ)
end

function compute_fisher!(
    F,
    ψ,
    π::StatelessSoftmaxPolicy
)
    fill!(F, 0.0)
    # Add a small constant to ensure stability
    F .= F + I*1e-4
    # For each probability of an action
    for a in 1:length(π.probs)
        # Calculate (and update) ψ vector of the log probability action gradients, under the
        # current policy π
        gradient_logp!(ψ, π, a)
        # Update the Fisher Information matrix, which represents how much from the sample (bandit history)
        # can be explained about the parameters calculated (i.e., the weights of the policy theta);
        # The Fisher Information matrix is calculated as a summation of the square of the gradient of the log
        # probability of the action a, multiplied by the probability of the action given the policy;
        F .+= π.probs[a] .* (ψ * ψ')
    end
end

function compute_fisher!(
    F,
    ψ,
    π::StatelessNormalPolicy
)
    fill!(F, 0.0)
    N = length(π.σ)
    F[diagind(F)] .= vcat(π.σ..., 2.0*π.σ...)
end

"""

idxs is the array of training timesteps from the bandit history.
ϕ is the basis function that translates an integer (timestep) into a feature vector.

"""
function maximize_nsbs_lower!(
    params, 
    π::StatelessSoftmaxPolicy, 
    D, 
    idxs, 
    ϕ, 
    τ, 
    num_boot, 
    δ, 
    aggf, 
    λ, 
    IS, 
    old_ent, 
    num_iters, 
    rng
)
    # Get the theta parameters used to model the policy
    θ = get_params(π)
    # Prepare the function that is used to perform the off-policy natural gradient bootstrap method
    g! = nsbs_lower_grad(
        θ, 
        D, 
        idxs, 
        π, 
        ϕ, 
        τ,
        # This is the number of train samples used for bootstrapping the confidence interval
        num_boot, 
        δ, 
        aggf, 
        λ, 
        IS, 
        old_ent, 
        rng
    )
    # Optimize, for the number of optimization stepd, the parameters of the policy
    result = optimize(
        params, 
        g!, 
        θ, 
        num_iters
    )

    # Set the updated policy parameters into the proper vector
    @. θ = result
    # Update the policy data structure by setting the new parameters
    set_params!(π, θ)
end


"""
This function is used to build the natural policy gradient function and the confidence interval function.

ϕ: the basis function that translates an integer (timestep) into a feature vector.
τ: the number of future timesteps to consider.
nboot_train: the number of train samples used for bootstrapping the confidence interval.
nboot_test: the number of test samples used for bootstrapping the confidence interval.
δ: the percentile lower bound to maximize future.
λ: the entropy regularizer coefficient.
IS: the importance sampling method used to estimate the expected return of a given policy 
    on samples which were drawn using a different policy.
old_ent: a boolean flag that indicates whether to use the old entropy calculation method.
num_iters: the number of times the optimization must be run, calculated before as a percentage (τ*opt_ratio).
rng: the random number generator used to generate random numbers.
"""
function build_nsbst(
    ϕ, 
    # Note: ";" is used to separate positional arguments from keyword arguments
    τ; 
    nboot_train=200, 
    nboot_test=500, 
    δ=0.05, 
    λ=0.01, 
    IS=PerDecisionImportanceSampling(),
    # TODO MSG what does old_ent mean?
    old_ent=false, 
    num_iters=100, 
    rng=Base.GLOBAL_RNG
)
    opt_fun(oparams, π, D, idxs) = maximize_nsbs_lower!(
        oparams, 
        π, 
        D, 
        idxs, 
        ϕ, 
        τ, 
        nboot_train, 
        δ, 
        mean, 
        λ, 
        IS, 
        old_ent, 
        num_iters, 
        rng
    )
    bound_fun(D, idxs, π, δ, tail) = nswildbst_CI(
        π, 
        δ, 
        tail, 
        D, 
        idxs, 
        ϕ, 
        τ, 
        nboot_test, 
        IS, 
        rng
    )
    return opt_fun, bound_fun
end


function off_policy_natgrad!(G, θ, Y, GY, ψ, F, π, D::BanditHistory{T,TA}, idxs, A, B, δ, num_boot, aggf, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)
    estimate_entropyreturn!(Y, D, idxs, π, λ, IS)
    GY .= Zygote.gradient(y->mean(get_preds_and_residual(y, A, B)[1]), Y)[1]
    fill!(G, 0.0)

    for (i,idx) in enumerate(idxs)
        gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end
    G ./= length(idxs)
    G .= inv(F) * G
end

function off_policy_natgrad!(G, θ, Y, GY, ψ, F, π::StatelessNormalPolicy, D::BanditHistory{T,TA}, idxs, A, B, δ, num_boot, aggf, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)

    estimate_entropyreturn!(Y, D, idxs, π, λ, IS)
    GY .= Zygote.gradient(y->mean(get_preds_and_residual(y, A, B)[1]), Y)[1]

    fill!(G, 0.0)
    for (i,idx) in enumerate(idxs)
        logp = gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end
    G ./= length(idxs)
    G .= inv(F) * G
end

function off_policy_natgrad_old!(G, θ, Y, GY, ψ, F, π::StatelessNormalPolicy, D::BanditHistory{T,TA}, idxs, A, B, δ, num_boot, aggf, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)

    estimate_return!(Y, D, idxs, π, IS)
    GY .= Zygote.gradient(y->mean(get_preds_and_residual(y, A, B)[1]), Y)[1]

    fill!(G, 0.0)
    G .= λ .* gradient_entropy(π)
    for (i,idx) in enumerate(idxs)
        logp = gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end
    G ./= length(idxs)
    G .= inv(F) * G
end

# TODO is this function even called?
function ns_lower_grad(θ₀, D::BanditHistory{T,TA}, idxs, π, ϕ, τ, num_boot, δ, aggf, λ, IS::TI, old_ent, rng) where {T,TA,TI<:UnweightedIS}
    L = length(D)
    N = length(idxs)
    Y = zeros(T, N)
    GY = similar(Y)
    ψ = similar(θ₀)

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B = get_coefs(Φ, ϕτ)

    F = zeros(T, (length(ψ), length(ψ)))

    if old_ent
        return (G, θ)->off_policy_natgrad_old!(G, θ, Y, GY, ψ, F, π, D, idxs, A, B, δ, num_boot, aggf, λ, IS, rng)
    else
        return (G, θ)->off_policy_natgrad!(G, θ, Y, GY, ψ, F, π, D, idxs, A, B, δ, num_boot, aggf, λ, IS, rng)
    end
end
