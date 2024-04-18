import Base:length,push!


# Define a struct to hold the history of a bandit problem.
# <: is the subtype operator, which is used to define a new type as a subtype of an existing type;
# The BanditHistory struct is a subtype of the Any type, which is the root of the type hierarchy in Julia;
# The BanditHistory struct has two type parameters, T and TA, which are used to specify the types of the 
# elements in the actions and blogps arrays;
struct BanditHistory{T,TA} <: Any where {T,TA}
    # Array to hold the actions taken at each step
    actions::Array{TA,1}
    
    # Array to hold the log-probabilities of the actions taken at each step
    blogps::Array{T,1}
    
    # Array to hold the rewards received at each step
    rewards::Array{T,1}

    # Constructor for the BanditHistory struct
    # ::Type{T} is a type assertion that specifies that the first argument to the constructor must be of type T;
    # ::Type{TA} is a type assertion that specifies that the second argument to the constructor must be of type TA;
    function BanditHistory(::Type{T}, ::Type{TA}) where {T,TA}
        # Initialize the actions, blogps, and rewards arrays as empty arrays
        new{T,TA}(Array{TA,1}(), Array{T,1}(), Array{T,1}())
    end
end

# Define a struct to hold the trajectory of a reinforcement learning problem
struct Trajectory{T,TS,TA} <: Any where {T,TS,TA}
    # Array to hold the states visited at each step
    states::Array{TS,1}
    
    # Array to hold the actions taken at each step
    actions::Array{TA,1}
    
    # Array to hold the log-probabilities of the actions taken at each step
    blogps::Array{T,1}
    
    # Array to hold the rewards received at each step
    rewards::Array{T,1}

    # Constructor for the Trajectory struct
    function Trajectory(::Type{T}, ::Type{TS}, ::Type{TA}) where {T,TS,TA}
        # Initialize the states, actions, blogps, and rewards arrays as empty arrays
        new{T,TS,TA}(Array{TS,1}(), Array{TA,1}(), Array{T,1}(), Array{T,1}())
    end
end

# Define a struct to hold the history of a reinforcement learning problem
struct History{T} <: Any where {T}
    # Array to hold the values of τ at each step
    τs::Array{T,1}

    # Constructor for the History struct
    function History(::Type{T}) where {T}
        # Initialize the τs array as an empty array
        new{T}(Array{T,1}())
    end
end

# Define a function to get the length of a BanditHistory rewards
function length(H::BanditHistory)
    return length(H.rewards)
end

# Define a function to get the length of a Trajectory object
function length(H::History)
    # Return the length of the τs array
    return length(H.τs)
end

# Define a function to get the length of a Trajectory object
function push!(H::History{T}, item::T) where {T}
    push!(H.τs, item)
end

# Define a function to add an action, log-probability, and reward to a BanditHistory object
function push!(H::BanditHistory{T,TA}, action::TA, blogp::T, reward::T) where {T,TA}
    push!(H.actions, action)
    push!(H.blogps, blogp)
    push!(H.rewards, reward)
end

function log_eval(env, action, rng, sample_counter, rec, eval_fn, π, πsafe)
    # Add one to the value that stores the current timestamp
    sample_counter[1] += 1
    record_perf!(rec, eval_fn, sample_counter[1], π, πsafe)
    # Return the sampled reward from the environment, given the current action
    return sample_reward!(env, action, rng)
end

"""
Collect data for a bandit problem by taking N actions using the policy π,
sampling rewards from the bandit problem using the sample_fn! function, and storing the actions,
log-probabilities, and rewards in the given BanditHistory object H.
"""
function collect_data!(H::BanditHistory, π, sample_fn!, N, rng)
    # For N timestep
    for i in 1:N
        # Get an action's log-probability from the policy π in a random manner
        logp = get_action!(π, rng)
        # Sample a reward from the bandit problem using the given function
        r = sample_fn!(π.action, rng)
        # Add the action, log-probability, and reward to the BanditHistory object H
        push!(H, deepcopy(π.action), logp, r)
    end
end
