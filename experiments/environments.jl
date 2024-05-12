abstract type AbstractBandit end
abstract type AbstractDiscreteBandit <: AbstractBandit end
abstract type AbstractContinuousBandit <: AbstractBandit end

"""
This data structure models the bandit for the Glucose experiment.


"""
struct NonstationaryQuadraticBanditParams{T} <: AbstractContinuousBandit where {T}
    a::T
    b::T
    c::T
    τ::T
    t::Array{T,1}
end

"""
This data structure models the bandit for the RecSys experiment.


"""
struct NonStationaryDiscreteBanditParams{T} <: AbstractDiscreteBandit where {T}
    # Mean reward of each action
    μ::Array{T,1}
    # Standard deviation of each action
    σ::Array{T,1}
    # Seasonal frequency of each action
    τ::Array{T,1}
    # Phase shift of each action
    k::Array{T,1}
    # Timestamp of the bandit problem
    t::Array{T,1}
end

# TODO what is this used for?
struct DiscreteBanditParams{T} <: AbstractDiscreteBandit where {T}
    μ::Array{T,1}
    σ::Array{T,1}
end

"""
This is the method used in the Glucose experiment.

"""
function sample_reward!(
    b::NonstationaryQuadraticBanditParams{T},
    action,
    rng
) where {T}
    r = -0.1*(b.a*action[1] + b.b*cos(b.t[1] * b.τ))^2 + b.c * randn(rng)
    b.t[1] += 1.0
    return r
end

"""
This is the method used in the RecSys experiment.

"""
function sample_reward!(
    b::NonStationaryDiscreteBanditParams{T}, 
    action::Int, 
    rng
) where {T}
    # Calculate the expected reward of a particular action in the bandit problem (multiplying it by the 
    # seasonal term), and add a noise to it to perturb the value of the performance;
    r = b.μ[action] * sin(b.t[1] * b.τ[action] + b.k[action]) + b.σ[action] * randn(rng)
    # Update the value of the timestamp in the bandit problem
    b.t[1] += 1.0
    return r
end

# TODO is this even called? There is no instantiation of this type.
function sample_reward!(
    b::DiscreteBanditParams{T}, 
    action::Int, 
    rng
) where {T}
    return b.μ[action] + b.σ[action] * randn(rng)
end

"""
This is the method used in the Glucose experiment.

"""
function eval_policy(
    b::NonstationaryQuadraticBanditParams{T},
    π::StatelessNormalPolicy
) where {T,T2}
    t = b.t[1]
    μ, σ = π.μ[1], π.σ[1]
    J = -0.1*b.a^2^σ^2  + -0.1*(b.a*μ + b.b*cos(t * b.τ))^2
    return J
end

"""
This is used in the RecSys experiment.

"""
# TODO cannot this be merged with the next method?
function eval_policy(
    b::T, 
    π::StatelessSoftmaxPolicy
) where {T<:AbstractDiscreteBandit}
    # Given the environment b and the policy π, evaluate the policy by taking the expected value of the rewards
    return eval_policy(b, π.probs)
end


"""
This is used in the RecSys experiment.


"""
function eval_policy(
    b::NonStationaryDiscreteBanditParams{T}, 
    p::Array{T2}
) where {T,T2}
    J = 0.0
    # Take the current timestep number, to properly calculate the return of the policy
    # based on the seasonality of the bandit problem
    t = b.t[1]
    # For each i between 1 and the length of the probabilities of taking any of the actions
    for i in 1:length(p)
        # Sum, to the total expected return, the expected return of action i under the current policy;
        # p = probability of the action;
        # b.μ = mean reward of the action;
        # b.τ = seasonal frequency of the action (same for all);
        # b.k = phase shift of the action (different for each action);
        J += p[i] * b.μ[i] * sin(t * b.τ[i] + b.k[i])
    end
    return J
end

# TODO is this even called? There is no instantiation of DiscreteBanditParams type.
function eval_policy(
    b::DiscreteBanditParams{T},
    p::Array{T2}
) where {T,T2}
    J = 0.0
    t = b.t[1]
    for i in 1:length(p)
        J += p[i] * b.μ[i]
    end
    return J
end
