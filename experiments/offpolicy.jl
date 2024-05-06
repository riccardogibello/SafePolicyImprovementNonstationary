using EvaluationOfRLAlgs

abstract type AbstractImportanceSampling end
abstract type UnweightedIS <: AbstractImportanceSampling end
abstract type WeightedIS <: AbstractImportanceSampling end

struct ImportanceSampling <: UnweightedIS end
struct PerDecisionImportanceSampling <: UnweightedIS end
struct WeightedImportanceSampling <: WeightedIS end
struct WeightedPerDecisionImportanceSampling <: WeightedIS end

function IS_weight(blogp, a, π)
    return exp(logprob(π, a) - blogp)
end

function IS_weight(blogp, logp)
    return exp(logp - blogp)
end

function estimate_entropyreturn!(
    G, 
    H::BanditHistory, 
    idxs, 
    π, 
    α,
    # Set a variable of this type, even if it is not used
    ::UnweightedIS
)
    # @. G = IS_weight(H.blogps, H.actions, (π,)) * H.rewards

    # For all the elements in idxs, get its index and the element itself
    for (i,t) in enumerate(idxs)
        # Calculate the current policy π log-probability of the action 
        # chosen by the behaviour policy at time t
        logp = logprob(π, H.actions[t])
        # Calculate the return estimate at time t of the current policy action
        # with the importance sampling method wrt the behaviour policy;
        # the return is regularized by the current action entropy, to prevent 
        # being stuck in a local minimum;
        G[i] = IS_weight(H.blogps[t], logp) * (H.rewards[t] - α * logp)
    end
end

function estimate_return!(G, H::BanditHistory, idxs, π, ::UnweightedIS)
    blogps = view(H.blogps, idxs)
    actions = view(H.actions, idxs)
    rewards = view(H.rewards, idxs)
    @. G = IS_weight(blogps, actions, (π,)) * rewards
end

function estimate_return!(G, H::BanditHistory, π, ::WeightedIS)
    W = @. IS_weight(H.blogps, H.actions, (π,))
    W ./= sum(W)
    @. G = W * H.rewards
end

function estimate_return!(G, H::History{T}, π, ::UnweightedIS) where {T<:Trajectory}
    println("error! estimating return not implemented for History of Trajectories")
end
