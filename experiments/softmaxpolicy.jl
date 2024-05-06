import EvaluationOfRLAlgs:AbstractPolicy, logprob, get_action!, gradient_logp!
import Base:length
using Random

function softmax(x)
    x = clamp.(x, -32., 32.)
    return exp.(x) / sum(exp.(x))
end

# function sample_discrete(p::Array{<:Real, 1}, rng)::Int
"""
Implement the inverse transform sampling method to sample a discrete probability distribution.
The inverse transform sampling method for generating random numbers that are distributed according
to a given probability distribution. This method is used to generate random numbers that follow
a specific distribution, rather than a uniform distribution. 
"""
function sample_discrete(p, rng)::Int
    # Get the length of the array of probabilities
    n = length(p)
    # Initialize a counter to 1
    i = 1
    # Initialize the cumulative probability to the first element of the array
    c = p[1]
    # Generate a random number between 0 and 1
    u = rand(rng)
    # Find the inverse of the CDF of the random number (i.e., the index of the
    # array which satisfies P(X <= x) = u, where X is the random variable and x is the value of the random variable)
    while c < u && i < n
        c += p[i += 1]
    end

    return i
end

mutable struct StatelessSoftmaxPolicy{TP} <: AbstractPolicy where {TP}
    θ::Array{TP}
    action::Int
    probs::Array{TP,1}

    # The where clause indicates the input type is generic
    function StatelessSoftmaxPolicy(::Type{T}, num_actions::Int) where {T}
        # Create the arrays that are used to initialize the theta and the p
        # fields with the same size as the number of actions
        θ = zeros(T, num_actions)
        p = zeros(T, num_actions)
        new{T}(θ, 0, p)
    end
end

function logprob(π::StatelessSoftmaxPolicy, action)
    return log(π.probs[action])
end

function get_action_probabilities!(π::StatelessSoftmaxPolicy)
    π.probs .= softmax(π.θ)
end

function get_action!(π::StatelessSoftmaxPolicy, rng)
    # Set the current policy action index to the newly drawn one
    π.action = sample_discrete(π.probs, rng)
    # Find the value of the log probability of the action
    logp = log(π.probs[π.action])
    return logp
end

# NOTE: the ! at the end of the function name is a convention in Julia to indicate that the function 
# modifies its arguments in place. This is a convention and not a requirement of the language.
"""
Calculate the gradient of the log probability of the action with respect to the policy and the 
chosen action.

grad is a vector of the same size as the policy parameter vector θ, and it is used to store the
gradient of the log probability of the action with respect to the policy parameter vector θ.

π is the policy object that is used to calculate the gradient.

action is the index of the action that was chosen by the policy.
"""
function gradient_logp!(
    grad, 
    π::StatelessSoftmaxPolicy, 
    action::Int
)
    # Set the gradient vector to zero
    fill!(grad, 0.)
    # Find the value of the log probability of the given action, different
    # from the one which was chosen by the policy
    logp = log(π.probs[action])

    # Subtract, from each value of the gradient vector (i.e., derivative
    # with respect to the specific input action, set all to zero), 
    # the probabilities of all actions
    grad .-= π.probs
    # Add 1 to the gradient of the chosen action, since it is the one having
    # more influence on the output
    grad[action] += 1.

    return logp
end

function gradient_entropy!(grad, π::StatelessSoftmaxPolicy)
    # fill!(grad, 0.)

    # p1 = zeros(length(π.probs))
    @. grad = -π.probs * log(π.probs)
    H = sum(grad)  # entropy (p1 has negative in it)
    @. grad += π.probs * H
    return H
end

function gradient_entropy(π::StatelessSoftmaxPolicy)
    grad = similar(π.θ)
    gradient_entropy!(grad, π)
    return grad
end

function get_action_gradient_logp!(grad, π::StatelessSoftmaxPolicy, rng)
    logp = get_action!(π, rng)
    gradient_logp!(grad, π, π.action)
    return logp
end

function set_params!(π::StatelessSoftmaxPolicy{T}, θ) where {T}
    # The @. macro is used to broadcast the operation to all elements of the array;
    # this is equivalent to π.μ = θ, but it is more efficient.
    # Set the theta field of the policy to the value of the parameter θ
    @. π.θ = θ
    get_action_probabilities!(π::StatelessSoftmaxPolicy)
end

function get_params(π::StatelessSoftmaxPolicy{T}) where {T}
    return π.θ
end

function copy_params!(params::Array{T}, π::StatelessSoftmaxPolicy{T}) where {T}
    @. params = π.θ
end

function copy_params(π::StatelessSoftmaxPolicy{T}) where {T}
    return deepcopy(π.θ)
end

function add_to_params!(π::StatelessSoftmaxPolicy{T}, Δθ) where {T}
    @. π.θ += Δθ
end

function clone(π::StatelessSoftmaxPolicy{T})::StatelessSoftmaxPolicy{T} where {T}
    π₂ = StatelessSoftmaxPolicy(T, length(π.θ))
    set_params!(π₂, π.θ)
    π₂.action = deepcopy(π.action)
    return π₂
end
