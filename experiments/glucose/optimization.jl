module optimization

include("../nonstationary_pi.jl")
using .nonstationary_pi:build_nsbst

include("glucose_env.jl")
using .glucose_env:sample_reward!

function optimize_nsglucose_safety(
    num_episodes, 
    rng, 
    speed, 
    hyperparams, 
    fpath, 
    save_res,
)
    # Create an history object to keep track of all the actions taken, 
    # their log-probabilities, and the rewards received
    D = BanditHistory(Float64, Array{Float64,1})
    # Instantiate the continuous glucose environment with the given non-stationarity speed
    # and a value for the seed to be given to the FDA simulator
    env = NSGlucoseSim(speed, abs(rand(rng, UInt32)))

    # Instantiate an initial policy, which is a normal policy with two actions
    # and a standard deviation of 2.0 for each action
    π = StatelessNormalPolicy(Float64, 2, 2.0, true)

    # Set the initial parameters for the policy and the softmax policy
    θ = [20.0, 30.0, 2.0, 2.0]
    set_params!(π, θ)
    πsafe = clone(π)

    # Set the function to draw rewards from the environment given an action and a random seed
    env_fn(action, _) = sample_reward!(env, action)
    # Set the function to collect data from the environment using the policy π
    sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)

    # Create the parameters for the Adam optimizer
    oparams = AdamParams(
        # Get the Adam optimization parameters 
        # (π.μ and π.σ if both are trainable, only π.μ otherwise)
        get_params(π), 
        1e-2; 
        β1=0.9, 
        β2=0.999, 
        ϵ=1e-5
    )
    # Unpack the given hyperparameters
    τ, λ, opt_ratio, fborder, old_ent = hyperparams

    # Set the number of bootstraps for the training and testing phases
    nboot_train = 200
    nboot_test = 500

    # Set the δ percentile lower bound to maximize future for (use 1-ϵ for upper bound)
    δ = 0.05
    IS = PerDecisionImportanceSampling()

    # Set the number of times the optimization step is run on the policy parameters
    num_opt_iters = round(Int, τ*opt_ratio)
    # Set the number of preliminary steps in which the agent will interact with the environment
    warmup_steps = 20
    sm = SplitLastKKeepTest(0.5)
    fb = fourierseries(Float64, fborder)
    nt = normalize_time(D, τ)
    ϕ(t) = fb(nt(t))

    # Calculate the number of iterations to run the policy optimization algorithm
    num_iters = num_episodes / τ

    train_idxs = Array{Int, 1}()
    test_idxs = Array{Int, 1}()

    # Build the functions to optimize the policy and perform the safety test
    opt_fun, safety_fun = build_nsbst(
        ϕ, 
        τ; 
        nboot_train=nboot_train, 
        nboot_test=nboot_test, 
        δ=δ, 
        λ=λ, 
        IS=IS, 
        old_ent=old_ent,
        num_iters=num_opt_iters,
        rng=rng
    )

    # Perform the process of High Confidence Off-Policy Improvement (HICOPI) by using the given
    # optimization and safety functions
    tidx, piflag = HICOPI!(
        oparams, 
        π, 
        D, 
        train_idxs, 
        test_idxs, 
        sample_fn, 
        opt_fun,
        safety_fun,
        πsafe,
        τ,
        δ,
        sm,
        num_iters,
        warmup_steps
    )

    # TODO change this to comply with the storage of the results
    # res = save_results(fpath, rec, D.rewards, tidx, piflag, save_res)
    res = save_results(fpath, D.rewards, tidx, piflag, save_res)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    return res
    # return (rec, D.rewards, tidx, piflag)
end

# TODO check this description
"""
This method performs the Non-Stationary Contextual Bandit optimization process for the Glucose environment.
In this case, the agent is advised to avoid actions that could lead to particulary dangerous
situations, such as hypoglycemia or hyperglycemia.
"""
function optimize_nscbandit_safety(
    num_episodes,
    rng::MersenneTwister,
    speed,
    hyperparams,
    fpath,
    save_res
)
    D = BanditHistory(Float64, Array{Float64,1})
    # Based on the speed, set the value of the κ parameter for seasonality of
    # the environment
    if speed == 0
        κ = 0.0
    elseif speed == 1
        κ = (2*pi)/2000.0
    elseif speed == 2
        κ = (2*pi)/1500.0
    elseif speed == 3
        κ = (2*pi)/1250.0
    else
        println("speed $speed not recognized")
        return nothing
    end

    # TODO what does it mean this type of environment?
    # Set the current environment
    env:: NonstationaryQuadraticBanditParams = NonstationaryQuadraticBanditParams(
        1.0,
        -10.0,
        0.010,
        κ,
        [1.0]
    )
    # env = NSGlucoseSim(speed, rand(Int,rng))

    # Set the initial parameters for the policy
    θ = zeros(2)
    θ[1] = 8
    θ[2] = 2

    # Create the policy, with a sigma that is actually trainable
    π = StatelessNormalPolicy(Float64, 1, 1, true)
    set_params!(π, θ)
    πsafe = clone(π)

    # Set the function to draw rewards from the environment given an action and a random seed
    env_fn(action, rng) = sample_reward!(env, action, rng)
    # Set the function to evaluate the policy
    eval_fn(π) = eval_policy(env, π)
    
    sample_counter = [0]
    rec = SafetyPerfRecord(Int, Float64)
    
    function log_eval(action, rng, sample_counter, rec, eval_fn, π, πsafe)
        sample_counter[1] += 1
        record_perf!(rec, eval_fn, sample_counter[1], π, πsafe)
        println(env.t[1], " ", round(env.b*cos(env.t[1] * env.τ), digits=3), " ", round.(get_params(π), digits=3))
        return env_fn(action, rng)
    end
    log_fn = (action, rng) -> log_eval(action, rng, sample_counter, rec, eval_fn, π, πsafe)

    # sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)
    sample_fn(D, π, N) = collect_data!(D, π, log_fn, N, rng)
    # Set the parameters for the optimization task
    oparams = AdamParams(get_params(π), 1e-2; β1=0.9, β2=0.999, ϵ=1e-5)
    # Unpack the set of hyperparameters
    τ, λ, opt_ratio, fborder, old_ent = hyperparams

    # Set the number of bootstraps for the training and testing phases
    nboot_train = 200
    nboot_test = 500
    # Set the δ percentile lower bound to maximize future for (use 1-ϵ for upper bound)
    δ = 0.05
    IS = PerDecisionImportanceSampling()

    # Set the number of times the optimization step is run on the policy parameters
    num_opt_iters = round(Int, τ*opt_ratio)
    # Set the number of preliminary steps in which the agent will interact with the environment
    warmup_steps = 20
    sm = SplitLastKKeepTest(0.5)
    fb = fourierseries(Float64, fborder)
    nt = normalize_time(D, τ)
    ϕ(t) = fb(nt(t))

    # Calculate the number of iterations to run the policy optimization algorithm
    num_iters = num_episodes / τ

    train_idxs = Array{Int, 1}()
    test_idxs = Array{Int, 1}()

    # Build the functions to optimize the policy and perform the safety test
    opt_fun, safety_fun = build_nsbst(
        ϕ, 
        τ; 
        nboot_train=nboot_train, 
        nboot_test=nboot_test, 
        δ=δ, 
        λ=λ, 
        IS=IS, 
        old_ent=old_ent, 
        num_iters=num_opt_iters, 
        rng=rng
    )
    # Perform the process of High Confidence Off-Policy Improvement (HICOPI) by using the given
    # optimization and safety functions
    tidx, piflag = HICOPI!(
        oparams, 
        π, 
        D, 
        train_idxs, 
        test_idxs, 
        sample_fn, 
        opt_fun, 
        safety_fun,
        πsafe,
        τ,
        δ,
        sm,
        num_iters,
        warmup_steps
    )

    # TODO refactor this
    # res = save_results(fpath, rec, D.rewards, tidx, piflag, save_res)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    # return res
    return (rec, D.rewards, tidx, piflag)
end

function optimize_disc_nsglucose_safety(
    num_episodes, 
    rng, 
    speed, 
    hyperparams, 
    fpath, 
    save_res, 
    seed
)
    # Create an history object to keep track of all the actions taken, 
    # their log-probabilities, and the rewards received
    D = BanditHistory(Float64, Int)
    # env = NSDiscreteGlucoseSim(speed, abs(rand(rng, UInt32)))

    # Set the initial probabilities for each action
    p = [0.5, 0.125, 0.125, 0.125, 0.125]
    # Normalize the probabilities to avoid overflow/underflow 
    # when dealing with probabilities
    θ = log.(p) .- mean(log.(p))
    # Retransform the θ vector to a vector of probabilities
    θ .= exp.(θ) / sum(exp.(θ))

    # Assign the θ vector to the π policy (and initialize the probabilities
    # to the softmax of the θ vector)
    π = StatelessSoftmaxPolicy(Float64, length(p))
    set_params!(π, θ)
    πsafe = clone(π)

    # Load the evaluation data for the glucose environment
    # from the CSV files in the "glucose_eval_data" directory
    mats, eval_means = load_eval(speed)

    # Set the current timestep counter to 0
    sample_counter = [0]
    
    rec = SafetyPerfRecord(Int, Float64)

    # Define a function that, taking a policy as input and the means of the 
    # returns of different actions, evaluates the expected return
    # of the policy π
    eval_fn(π) = eval_policy(sample_counter[1], π, eval_means)
    # Create an anonymous (lambda) function to be called inside the collect_data loop; this is used
    # to update the data structures of the bandit given a chosen action (both following the
    # pi and pi_safe policies);
    log_fn = (action, seed) -> log_eval(
        mats, 
        action, 
        seed, 
        sample_counter, 
        rec, 
        eval_fn, 
        π, 
        πsafe
    )
    # Define a sample_fn function to collect, given an environment, a policy, and a number of samples,
    # the data for the bandit problem
    sample_fn(D, π, N) = collect_data!(D, π, log_fn, N, seed)

    # Set the initial parameters for the optimization algorithm
    oparams = AdamParams(get_params(π), 1e-2; β1=0.9, β2=0.999, ϵ=1e-5)

    # Unpack the given hyperparameters
    τ, λ, opt_ratio, fborder, old_ent = hyperparams

    # Set the number of bootstraps for the training and testing phases
    nboot_train = 200
    nboot_test = 500
    # Set the δ percentile lower bound to maximize future (use 1-ϵ for upper bound)
    δ = 0.05
    IS = PerDecisionImportanceSampling()

    # Set the number of times the optimization step is run on the policy parameters
    # as the multiplication between the interaction steps (of each iteration, 
    # i.e., num_episodes / τ) and the optimization ratio
    num_opt_iters = round(Int, τ*opt_ratio)
    # Set a number of preliminary steps in which the agent will interact with the environment
    # prior to the safety optimization process
    warmup_steps = 20
    # Fraction of data samples to be used for training
    sm = SplitLastKKeepTest(0.5)
    # Call fourierseries method, so that fb will be a function to transform a scalar to a vector
    # of the cosine values of the products between the scalar and the elements of an incremental array C
    fb = fourierseries(Float64, fborder)
    # Create nt, a function that takes a timestep value and normalizes it with respect to the
    # length of the bandit history and the tau value (future steps)
    nt = normalize_time(D, τ)
    # Define the phi function, which accepts a timestep value, normalizes and fourier-transform it
    ϕ(t) = fb(nt(t))

    # Calculate the number of iterations to run the policy optimization algorithm
    # as the total number of timesteps to be run divided by tau (i.e., the number of
    # timesteps added at each iteration)
    # TODO shouldn't num_iters be an integer?
    # Approximate the num_iters to the nearest (upper) integer
    num_iters = round(Int, num_episodes / τ) + 1

    train_idxs = Array{Int, 1}()
    test_idxs = Array{Int, 1}()

    # Build the functions that are used to optimize the policy and perform the
    # safety test
    opt_fun, safety_fun = build_nsbst(
        ϕ, 
        τ; 
        nboot_train=nboot_train, 
        nboot_test=nboot_test, 
        δ=δ, 
        λ=λ, 
        IS=IS, 
        old_ent=old_ent, 
        num_iters=num_opt_iters, 
        rng=rng
    )

    # Perform the High Confidence Off-Policy Improvement (HICOPI) algorithm by using the given
    # optimization and safety functions
    tidx, piflag = HICOPI!(
        oparams, 
        π, 
        D, 
        train_idxs, 
        test_idxs, 
        sample_fn, 
        opt_fun, 
        safety_fun, 
        πsafe, 
        τ, 
        δ, 
        sm, 
        num_iters, 
        warmup_steps
    )

    res = save_results(
        fpath,
        rec,
        D.rewards,
        tidx,
        piflag,
        save_res
    )
    # res = save_results(fpath, D.rewards, tidx, piflag, save_res)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    return res
    # return (rec, D.rewards, tidx, piflag)
end

end