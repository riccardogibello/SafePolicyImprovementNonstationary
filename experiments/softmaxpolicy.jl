import EvaluationOfRLAlgs:AbstractPolicy, logprob, get_action!, gradient_logp!
import Base:length
using Random

"""
This is the definition of the StatelessSoftmaxPolicy type, which is used to model a policy
that uses the softmax function to calculate the probabilities of each action. Being stateless,
the action selection doesn't depend on any previous actions or states, 
it only depends on the current state.

θ: is the parameter vector of the policy.
action: is the index of the action that was chosen by the policy.
probs: is the array of softmax probabilities of each action of the policy.
"""
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

"""
This method returns the value of the log probability of the action given the policy.

π: Policy object used to calculate the log probability of the action.
action: Index of the action for which the log probability is calculated.
"""
function logprob(
    π::StatelessSoftmaxPolicy, 
    action
)
    return log(π.probs[action])
end

"""
This method is used to calculate the softmax function of the input array.
"""
function softmax(x)
    # Apply the clamping function to each element of the input array
    # to be contained in the range [-32, 32] (to avoid overflow when
    # calculating the exponential function of the input array)
    x = clamp.(x, -32., 32.)
    # Calculate the probability of each element of the array
    return exp.(x) / sum(exp.(x))
end

"""
This method populates the probabilities of each action given the policy
in the policy itself.

π: Policy object used to calculate the probabilities of each action.
"""
function get_action_probabilities!(
    π::StatelessSoftmaxPolicy
)
    π.probs .= softmax(π.θ)
end

"""
This method implements the inverse transform sampling method to sample a discrete probability distribution.
The inverse transform sampling method for generating random numbers that are distributed according
to a given probability distribution. This method is used to generate random numbers that follow
a specific distribution, rather than a uniform distribution.

p: is the array of probabilities of the actions of a discrete probability distribution.
rng: is the random number generator used to generate the random number.
"""
function sample_discrete(
    p::Array{<:Real,1},
    rng
)::Int
    # Get the length of the array of probabilities
    n = length(p)
    # Initialize a counter to 1
    i = 1
    # Initialize the cumulative probability to the first element of the array
    c = p[1]
    # Generate a random number between 0 and 1
    u = rand(rng)
    # Find the inverse of the CDF of the random number (i.e., the index of the
    # array which satisfies P(X <= x) = u, where X is the random variable and 
    # x is the value of the random variable)
    while c < u && i < n
        # Update the i counter and then access the i-th element of the array
        # of probabilities to update the cumulative probability
        c += p[i += 1]
    end

    return i
end

"""
This method is used to draw stochastically an action from the policy.

π: Policy object used to draw the action.
rng: Random number generator.
"""
function get_action!(
    π::StatelessSoftmaxPolicy,
    rng
)
    # Set the current policy action index to the newly drawn one
    π.action = sample_discrete(π.probs, rng)
    # Find the value of the log probability of the action
    logp = log(π.probs[π.action])
    return logp
end

"""
Calculate the gradient of the log probability of the action with respect to the policy and the 
chosen action.

grad: a vector of the same size as the policy parameter vector θ, and it is used to store the
gradient of the log probability of the action with respect to the policy parameter vector θ.

π: the policy object that is used to calculate the gradient.

action: the index of the action that was chosen by the policy.
"""
# NOTE: the ! at the end of the function name is a convention in Julia to indicate that the function 
# modifies its arguments in place. This is a convention and not a requirement of the language.
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

"""
This method calculates the gradient of the entropy of a StatelessSoftmaxPolicy 
and updates the grad array in-place.
The entropy is a measure of uncertainty or randomness in the policy. 
It quantifies the amount of "exploration" in the policy, with higher entropy 
corresponding to more exploration (more randomness in action selection), 
and lower entropy corresponding to more exploitation 
(more deterministic action selection).

In a discrete environment, the entropy of a policy is defined as:

                H(π) = - ∑ π(a|s) log π(a|s)

where π(a|s) is the probability of taking action a in state s according to the policy π.
Then, the gradient is calculated as:

                ∇H(π) = -π.probs * log(π.probs) + π.probs * H

In Reinforcement Learning, the entropy of a policy is added to the objective function 
to encourage exploration.

grad: a vector (of the same size as the policy parameter vector θ) that stores 
the gradient of the entropy of the policy with respect to its parameters.
"""
function gradient_entropy!(
    grad,
    π::StatelessSoftmaxPolicy
)
    # Calculate the entropy of each action
    @. grad = -π.probs * log(π.probs)
    # Calculate the total entropy
    H = sum(grad)

    # Add the contribution of each action to the total gradient of the entropy
    # (higher for more probable actions)
    @. grad += π.probs * H

    # Return the value of the total entropy
    return H
end


"""
This method calculates a vector containing the gradient of the entropy of the policy
with respect to the policy parameters. This is used to update the policy parameters
in order to optimize the objective function and encourage exploration.

π: the policy object used to calculate the gradient of the entropy.
"""
function gradient_entropy(
    π::StatelessSoftmaxPolicy
)
    # Build a gradient vector of the same size as the policy parameter vector θ
    grad = similar(π.θ)
    # Calculate the gradient of the entropy of the policy and of each action
    gradient_entropy!(grad, π)
    # Return the vector containing the gradients of each action
    return grad
end

"""
This method calculates the gradient of the log-probability of the selected action 
with respect to the policy parameters, and stores the result in grad. 
After this function, each element in grad corresponds to the derivative of 
the log-probability of the selected action with respect to a particular policy 
parameter.
"""
function get_action_gradient_logp!(
    grad,
    π::StatelessSoftmaxPolicy,
    rng
)
    # Get the log probability of a randomly sampled action
    logp = get_action!(π, rng)
    # Update the grad vector, containing the gradients of the softmax action 
    # probabilities
    gradient_logp!(grad, π, π.action)
    # Return the value of the log probability of the action
    return logp
end

"""
This method sets the parameters of the policy to the value of the parameter θ.
Moreover, it updates the probabilities of each action given the policy, calculating
the softmax function of the parameter vector θ.

π: the policy object used to set the parameters.
θ: the parameter vector used to set the parameters of the policy.
"""
function set_params!(
    π::StatelessSoftmaxPolicy{T},
    θ
) where {T}
    # The @. macro is used to broadcast the operation to all elements of the array;
    # this is equivalent to π.μ = θ, but it is more efficient.
    # Set the theta field of the policy to the value of the parameter θ
    @. π.θ = θ
    get_action_probabilities!(π::StatelessSoftmaxPolicy)
end

"""
This method returns the parameters of the policy.

π: the policy object used to get the parameters.
"""
function get_params(
    π::StatelessSoftmaxPolicy{T}
) where {T}
    return π.θ
end

"""
This method clones the policy object and returns a new policy object with the same
parameters as the original policy.

π: the policy object used to clone the policy.
"""
function clone(
    π::StatelessSoftmaxPolicy{T}
)::StatelessSoftmaxPolicy{T} where {T}
    π₂ = StatelessSoftmaxPolicy(T, length(π.θ))
    set_params!(π₂, π.θ)
    π₂.action = deepcopy(π.action)
    return π₂
end
