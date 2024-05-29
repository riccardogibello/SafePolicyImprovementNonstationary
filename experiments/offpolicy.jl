using EvaluationOfRLAlgs

include("history.jl")

abstract type AbstractImportanceSampling end
abstract type UnweightedIS <: AbstractImportanceSampling end
abstract type WeightedIS <: AbstractImportanceSampling end

struct ImportanceSampling <: UnweightedIS end
struct PerDecisionImportanceSampling <: UnweightedIS end
struct WeightedImportanceSampling <: WeightedIS end
struct WeightedPerDecisionImportanceSampling <: WeightedIS end

"""
This method computes the importance sampling weight of a policy π
with respect to the behaviour policy π_b, given the action a taken
by the behaviour policy (and the related log-probability blogp).

blogp: is the log-probability vector of the actions taken by the behaviour policy.
a: is the vector of actions taken by the behaviour policy.
π: is the policy for which the importance sampling weight is computed.
"""
function IS_weight(blogp, a, π)
    return exp(logprob(π, a) - blogp)
end

"""
This method computes the importance sampling weight of a policy π
with respect to the behaviour policy π_b.

blogp: is the log-probability of the actions taken by the behaviour policy.
logp: is the log-probability of the action taken by the policy π.
"""
function IS_weight(blogp, logp)
    return exp(logp - blogp)
end


"""
This function calculates for each timestamp of the training dataset
the estimated return of the current policy, using the importance sampling method
on the returns of the behaviour policy.

G: is the array of importance-sampled return estimates.
H: is the history of the bandit problem, containing the 
rewards obtained with the actions taken by the behaviour policy.
idxs: is the array of train indexes (see "HICOPI_step").
π: is the currently computed policy.
λ: is the entropy regularization coefficient.
"""
function estimate_entropyreturn!(
    G, 
    H::BanditHistory, 
    idxs, 
    π, 
    λ,
    # Set a variable of this type, even if it is not used
    ::UnweightedIS
)
    # @. G = IS_weight(H.blogps, H.actions, (π,)) * H.rewards

    # For all the elements in idxs, get its index and the element itself
    for (i,t) in enumerate(idxs)
        # Calculate the current policy π log-probability of the action 
        # chosen by the behaviour policy at time t
        logp = logprob(π, H.actions[t])
        # Calculate the return estimate at time t of the current π policy as
        # the product between:
        # 1) the importance sampling weight of the current π policy action
        # 2) the return of the behaviour policy at time t, subtracted the 
        # entropy regularization term to avoid local minima (multiplied 
        # by the coefficient λ to balance eploration and exploitation).
        G[i] = IS_weight(H.blogps[t], logp) * (H.rewards[t] - λ * logp)
        # NOTE: The total entropy of the policy is not used, but only the
        # log-probability of the π policy action is used as a proxy of the entropy
        # (for computational efficiency); the effect is the same, because
        # actions that are unlikely to be chosen (low probability) have a high
        # entropy value added to the return estimate (inducing exploration).
    end
end

function estimate_return!(
    G,
    H::BanditHistory,
    idxs,
    π,
    ::UnweightedIS
)
    blogps = view(H.blogps, idxs)
    actions = view(H.actions, idxs)
    rewards = view(H.rewards, idxs)
    # Perform the importance sampling estimate of the return
    # based on the log-probabilities of the history,
    # the actions taken by the behaviour policy, and the newly
    # computed policy.
    @. G = IS_weight(blogps, actions, (π,)) * rewards
end

# TODO this should not be called because a subclass of WeightedIS is never instantiated
function estimate_return!(
    G,
    H::BanditHistory,
    π,
    ::WeightedIS
)
    W = @. IS_weight(H.blogps, H.actions, (π,))
    W ./= sum(W)
    @. G = W * H.rewards
end

function estimate_return!(
    G,
    H::History{T},
    π,
    ::UnweightedIS
) where {T<:Trajectory}
    println("error! estimating return not implemented for History of Trajectories")
end
