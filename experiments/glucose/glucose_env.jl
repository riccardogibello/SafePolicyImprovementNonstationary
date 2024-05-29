module glucose_env

using PyCall
include("../environments.jl")

gsim = pyimport("simglucose")

struct NSGlucoseSim <: AbstractContinuousBandit
    env::PyObject

    function NSGlucoseSim(speed::Int, seed)
        # Set the current glucose environment by instantiating the Python FDA implementation
        new(gsim.envs.NS_T1DSimEnv(speed=speed, oracle=-1, seed=seed))
    end
end
export NSGlucoseSim

struct NSDiscreteGlucoseSim <: AbstractDiscreteBandit
    env::PyObject
    actions::Array{Tuple{Float64,Float64},1}

    function NSDiscreteGlucoseSim(speed::Int, seed)
        env = gsim.envs.NS_T1DSimEnv(speed=speed, oracle=-1, seed=seed)
        actions = Array{Tuple{Float64,Float64},1}([(23.0,33.0,), (19.0,26.0), (14.0,21.0), (9.0,16.0), (4.0,11.0)])
        new(env, actions)
    end
end

function sample_reward!(
    b::NSDiscreteGlucoseSim, 
    action::Int, 
    rng=nothing
) where {T}
    b.env.reset()
    _, r, _, _ = b.env.step(b.actions[action])
    return convert(Float64,r)
end

"""
This method receives the FDA simulator as input and returns the reward from the 
FDA simulator after taking the given action.

b: NSGlucoseSim - The FDA simulator.
action: Array{Float64,1} - The action to take in the FDA simulator.
"""
function sample_reward!(
    b::NSGlucoseSim, 
    action::Array{Float64,1}, 
)
    b.env.reset()
    # Get the reward from the environment by taking the action
    _, r, _, _ = b.env.step(action)
    # Return the reward as a Float64
    return convert(Float64,r)
end

"""
This is the method used in the Glucose experiment.

"""
function sample_reward!(
    b::NonstationaryQuadraticBanditParams{T},
    action,
    rng
) where {T}
    # Sample a reward given the current action
    r = -0.1 * ( b.a * action[1] + b.b * cos(b.t[1] * b.τ) )^2 + b.c * randn(rng)
    # Update the current timestamp in the bandit problem
    b.t[1] += 1.0
    return r
end

end