using Statistics
using EvaluationOfRLAlgs
using Random
using DataFrames
using CSV
using ArgParse
using Distributions
using Plots
using Profile
using ProfileView
using Base.Threads
using Distributed

using Logging
Logging.global_logger(SimpleLogger(stderr, Logging.Error))

# Set the number of processes to be used in the experiment
max_processes = 10
# The newly created workers will span from 2 to max_processes + 1
addprocs(max_processes)
println("Number of active processes: ", nprocs())

@everywhere begin
    function get_max_processes()
        return 11
    end
end


@everywhere include("normalpolicy.jl")
@everywhere include("softmaxpolicy.jl")

@everywhere include("optimizers.jl")
@everywhere include("offpolicy.jl")

@everywhere include("history.jl")

@everywhere include("environments.jl")
@everywhere include("nonstationary_modeling.jl")
@everywhere include("nonstationary_pi.jl")

function plot_results(rec::SafetyPerfRecord, rews, tidxs, piflag, title)
    p1 = plot(title=title)
    p2 = plot()
    #p3 = plot()
    issafety = zeros(length(rews))
    unsafe = zeros(length(rews))
    notnsf = zeros(length(rews))
    for (i, (ts, te)) in enumerate(tidxs)
        color = :crimson
        if piflag[i]
            color = :dodgerblue
            notnsf[ts:te] .= 1
            if mean(rec.Jsafe[ts:te]) > mean(rec.Jpi[ts:te])
                unsafe[ts:te] .= 1
            end
        end
        scatter!(p1, ts:te, rews[ts:te], markercolor=color, markersize=1, markerstrokewidth=0, label=nothing)
        # issafety[ts:te] .= piflag[i]
    end
    # rgrad = cgrad([:crimson, :dodgerblue])
    # plot!(p1, rec.t, rews, linestyle=:dot, lc=rgrad, line_z=issafety, label="observed rewards")
    plot!(p1, rec.t, rec.Jsafe, lc=:crimson, label="J(π_safe)")
    plot!(p1, rec.t, rec.Jpi, lc=:dodgerblue, label="J(π_c)", legend=:bottomleft)
    # plot!(p2, )
    xlabel!(p1, "Episode")
    ylabel!(p1, "Return")

    plot!(p2, rec.t, notnsf, label="Canditate Returned")
    plot!(p2, rec.t, unsafe, label="Unsafe Policy", legend=:bottomright)
    xlabel!(p2, "Episode")
    ylabel!(p2, "Probability")
    p = plot(p1, p2, layout=(2,1))
    savefig(p, "myplot.pdf")
    return p
end



# TODO ns is a bit misleading since it is called also for non-stationary problems, isn't it?
@everywhere function optimize_nsdbandit_safety(
    num_episodes,
    rng,
    speed,
    hyperparams,
)
    # Create a new BanditHistory object to store the history of the bandit problem
    D = BanditHistory(Float64, Int)
    # Set the payoffs for each of the actions of the bandit problem
    arm_payoffs = [1.0, 0.8, 0.6, 0.4, 0.2]
    # Set the array of frequencies for each action of the bandit problem
    arm_freq = zeros(length(arm_payoffs))
    # if the problem is stationary
    if speed == 0
        # set the value that modifies the frequency of the sin seasonal term to 0
        κ = 0
    # otherwise, set an increasing frequency of the sin seasonal term as the speed increases
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
    # Set the frequency of all the actions to the value stored in κ
    arm_freq .= κ
    # Create a new arm_k array, uninitialized, with the same length as the arm_freq array
    arm_k = similar(arm_freq)
    # Assign to each element of arm_k the value of pi/2
    arm_k .= pi/2
    # Add to each element of the arm_k array the element-wise product between 2*pi/5 and the array on the right;
    # arm_k stores a different seasonal term for each of the bandit actions
    arm_k .+= (2*pi/5) .* [0.0, 1.0, 2.0, 3.0, 4.0]
    # Create a new array of ones with the same length as the arm_payoffs array
    # that represents the noise for each arm payoff
    arm_sigma = ones(length(arm_payoffs))
    arm_sigma .*= 0.05

    # Create a new NonStationaryDiscreteBanditParams object with the arm_payoffs, arm_sigma, arm_freq, and arm_k arrays
    env:: NonStationaryDiscreteBanditParams = NonStationaryDiscreteBanditParams(
        # arm_payoffs contains the payoffs of each arm
        arm_payoffs,
        # arm_sigma contains the noise of each arm payoff
        arm_sigma,
        # arm_freq contains the seasonal frequency that affects the return of each action
        arm_freq,
        # arm_k contains the horizontal shifts of the seasonal sin term that affects the return of each action
        arm_k,
        # t contains the current timestep of the bandit problem
        [0.0]
    )

    # Set the theta values, that are the policy parameters, for each arm
    θ = [2.0, 1.5, 1.2, 1.0, 1.0]

    # Create a new policy pi to choose among the arm_payoffs
    π = StatelessSoftmaxPolicy(Float64, length(arm_payoffs))
    # Set the parameters of the policy to the θ values 
    # (and consequently, the action probabilities 
    # as a softmax of the θ values)
    set_params!(π, θ)
    # Clone the built policy and assign it to πsafe
    πsafe = clone(π)

    # Prepare a counter for the samples
    sample_counter = [0]
    # Instantiate a variable that contains the record of the performance of the policy
    rec = SafetyPerfRecord(Int, Float64)

    # Set the parameters for the optimization algorithm (Adam)
    oparams = AdamParams(get_params(π), 1e-2; β1=0.9, β2=0.999, ϵ=1e-5)
    # τ = number of steps for which data are collected
    τ, λ, opt_ratio, fb_order = hyperparams
    # Cast τ and fb_order to Int to avoid errors
    τ = Int(τ)
    fb_order = Int(fb_order)
    # Set the δ percentile lower bound to maximize future (use 1-ϵ for upper bound)
    δ = 0.05
    IS = PerDecisionImportanceSampling()

    nboot_train = 200 # num bootstraps
    nboot_test = 500
    # Set the number of times the optimization step is run on the policy parameters
    # as the multiplication between the interaction steps (of each iteration, 
    # i.e., num_episodes / τ) and the optimization ratio
    num_opt_iters = round(Int, τ * opt_ratio)
    # Set a number of preliminary steps in which the agent will interact with the environment
    # prior to the safety optimization process
    warmup_steps = 20
    sm = SplitLastKKeepTest(0.25)  # Fraction of data samples to be used for training
    # Create nt, a function that takes a timestep value and normalizes it with respect to the
    # length of the bandit history and the tau value (future steps)
    nt = normalize_time(D, τ)
    # Call fourierseries method, so that fb will be a function to transform a scalar to a vector
    # of the cosine values of the products between the scalar and the elements of an incremental array C
    fb = fourierseries(Float64, fb_order)
    # Define the phi function, which accepts a timestep value, normalizes and fourier-transform it
    ϕ(t) = fb(nt(t))

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
        num_iters=num_opt_iters, 
        rng=rng
    )

    # ===========================================================
    # Define a function that, taking a policy as input, evaluates the expected return given the
    # environment (global variable or within the scope) and the policy
    eval_fn(π) = eval_policy(env, π)
    # Create an anonymous (lambda) function to be called inside the collect_data loop; this is used
    # to update the data structures of the bandit given a chosen action (both following the
    # pi and pi_safe policies);
    log_fn = (action, rng) -> log_eval(env, action, rng, sample_counter, rec, eval_fn, π, πsafe)
    # Define a sample_fn function to collect, given an environment, a policy, and a number of samples,
    # the data for the bandit problem
    sample_fn(D, π, N) = collect_data!(D, π, log_fn, N, rng)
    # ===========================================================
    # Calculate the number of iterations to run the policy optimization algorithm
    # as the total number of timesteps to be run divided by tau (i.e., the number of
    # timesteps added at each iteration)
    # TODO shouldn't num_iters be an integer?
    # Approximate the num_iters to the nearest (upper) integer
    num_iters = round(Int, num_episodes / τ) + 1
    train_idxs = Array{Int, 1}()
    test_idxs = Array{Int, 1}()
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

    # res = save_results(rec, D.rewards, tidx, piflag)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    # return res
    # TODO actually this is the needed return value for printing the learning curves
    return (rec, D.rewards, tidx, piflag)
end

@everywhere begin 
    """
        This method receives as input the results of the previous experiments 
        (i.e., the record data structures, the reward values from the bandit, 
        the start and end timesteps on any interaction of the agent with the environment (i.e. start - end = tau), 
        and the flags indicating if the policy computed was used against the safe one).


        The result given by the method is composed by timestamps, 
        the mean and standard deviation of the unsafe policy, the mean and standard deviation 
        of the found policy, the mean and standard deviation of the performance,
        and the mean and standard deviation of the performance of the algorithm.
    """
    function combine_trials(
        results
    )
        T = length(results)
        N = length(results[1][2])

        # Take the list of timestamps of the first experiment (they are the same for all the experiments)
        t = results[1][1].t
        
        # Initialize matrices with NxT dimension
        unsafe = zeros((N, T))
        notnsf = zeros((N, T))
        Jpi = zeros((N, T))
        Jalg = zeros((N, T))

        # For each of the results of the trials (k is the index of the trial)
        for (k, (rec, _, tidxs, piflag)) in enumerate(results)
            unf = zeros(N)
            nnsf = zeros(N)
            # For each starting and ending timestep in the tidxs array
            for (i, (ts, te)) in enumerate(tidxs)
                # If the computed new policy was used against the 
                # safe one (i.e., passed the safety test)
                if piflag[i]
                    # Set to 1 the values of the notnsf array from the start to the end timestep
                    nnsf[ts:te] .= 1
                    # If the mean return of the safe policy would have been higher than the one of the computed policy
                    if mean(rec.Jsafe[ts:te]) > mean(rec.Jpi[ts:te])
                        # Set to 1 the values of the unsafe array from the start to the end timestep
                        unf[ts:te] .= 1
                    end
                end
            end
            # Update the unsafe matrix by adding, as k-th column, the values of the unf array
            @. unsafe[:, k] = unf
            # Update the notnsf matrix by adding, as k-th column, the values of the nnsf array
            @. notnsf[:, k] = nnsf
            # Update the Jpi matrix by adding, as k-th column, the values of the Jpi array of the k-th trial
            @. Jpi[:, k] += rec.Jpi
            # Update the Jalg matrix by adding, as k-th column, the linear combination of the performances 
            # of the safe and computed policies (based on when each of them was adopted).
            @. Jalg[:, k] += nnsf * rec.Jpi + (1 - nnsf) * rec.Jsafe
        end

        # Perform the mean across the second dimension of each matrix (i.e., across experiments), creating
        # a set of 1D vectors (the standard deviation vectors are divided by the square root of the number of trials);
        # mnunsafe = % of times, for each timestep, that the computed policy was used against the safe one but had a lower return
        mnunsafe = vec(mean(unsafe, dims=2))
        stdunsafe = vec(std(unsafe, dims=2)) ./ √T
        # mnfound = % of times, for each timestep, that the computed policy was used against the safe one
        mnfound = vec(mean(notnsf, dims=2))
        stdfound = vec(std(notnsf, dims=2)) ./ √T
        # mnJpi = mean performance of the computed policy
        mnJpi = vec(mean(Jpi, dims=2))
        stdJpi = vec(std(Jpi, dims=2)) ./ √T
        # mnJalg = mean performance of the computed policy if it was used, otherwise the safe policy
        mnJalg = vec(mean(Jalg, dims=2))
        stdJalg = vec(mean(Jalg, dims=2)) ./ √T
        return t, (mnunsafe, stdunsafe), (mnfound, stdfound), (mnJpi, stdJpi), (mnJalg, stdJalg)
    end

    """
    This method plots, for the given results:
    1) the performances of the BASELINE and SPIN algorithms.
    2) the percentage of times across the experiments in which the computed policy was used against the safe one
    (i.e., passed the safety test) and had a lower return.
    3) the percentage of times across the experiments in which the computed policy was used against the safe one
    (i.e., passed the safety test).

    res1: the results of the set of experiments related to the BASELINE algorithm, which is not aware of the non-stationarity of the environment.
    res2: the results of the set of experiments related to the SPIN algorithm, which is aware of the non-stationarity of the environment.
    baseline: the performance of the π_safe policy.
    labels: the labels of the two algorithms.
    title: the title of the plot.
    """
    function learning_curves(
        # TODO shouldn't it be more clear to call these res_baseline and res_spin?
        res1, 
        res2,
        # TODO shouldn't it be more clear to call this π_safe_perf?
        baseline, 
        labels, 
        title,
        path,
    )
        p1 = plot(title=title)
        p2 = plot()

        # ============================== PLOT 1 ASSIGNMENTS ==============================
        # Unpack the results of the set of experiments related to the agent which is not aware of the 
        # non-stationarity of the environment
        t1, safe1, found1, Jpi1, Jalg1 = res1
        # Set the line that contains the performances of the π_safe policy (dotted black line)
        plot!(p1, t1, baseline, lc=:black, label="π_safe")
        # Set the line that contains the performance of the BASELINE algorithm (i.e., not accounting
        # for the non-stationarity of the environment, red line)
        plot!(p1, t1, Jalg1[1], ribbon=Jalg1[2], lc=:crimson, fillalpha=0.3, label=labels[1])
        # Set the line that contains the performances of the π_c policy of the BASELINE algorithm 
        # (dotted red line)
        plot!(p1, t1, Jpi1[1], lc=:crimson, linestyle=:dot, label=nothing)

        # Unpack the results of the set of experiments related to the agent which is aware of the 
        # non-stationarity of the environment
        t2, safe2, found2, Jpi2, Jalg2 = res2
        # Set the line that contains the performances of the SPIN algorithm (blue line)
        plot!(p1, t2, Jalg2[1], ribbon=Jalg2[2], lc=:dodgerblue, fillalpha=0.3, label=labels[2])
        # Set the line that contains the performances of the π_c policy of the SPIN algorithm (dotted blue line)
        plot!(p1, t2, Jpi2[1], lc=:dodgerblue, linestyle=:dot, label=nothing, legend=:topleft)
        
        # Set the x and y axis labels
        xlabel!(p1, "Episode")
        ylabel!(p1, "Performance")
        # ================================================================================

        # ============================== PLOT 2 ASSIGNMENTS ==============================
        # Plot, for the BASELINE, the percentage of times across the experiments in which the computed policy was used
        # against the safe one (i.e., passed the safety test) (red dotted line)
        plot!(p2, t1, found1[1], ribbon=found1[2], linestyle=:dash, lc=:crimson, fillalpha=0.3, label="Canditate Returned-Baseline")
        # Plot, for the BASELINE, the percentage of times across the experiments in which the computed policy was used
        # against the safe one (i.e., passed the safety test) but had a lower return (red line)
        plot!(p2, t1, safe1[1], ribbon=safe1[2], linestyle=:solid, lc=:crimson, fillalpha=0.3,  label="Unsafe Policy-Baseline")
        # Plot, for the SPIN algorithm, the percentage of times across the experiments in which the computed policy was used
        # against the safe one (i.e., passed the safety test) (blue dotted line)
        plot!(p2, t2, found2[1], ribbon=found2[2], linestyle=:dash, lc=:dodgerblue, fillalpha=0.3,  label="Canditate Returned-SPIN")
        # Plot, for the SPIN algorithm, the percentage of times across the experiments in which the computed policy was used
        # against the safe one (i.e., passed the safety test) but had a lower return (blue line)
        plot!(p2, t2, safe2[1], ribbon=safe2[2], linestyle=:solid, lc=:dodgerblue, fillalpha=0.3,  label="Unsafe Policy-SPIN", legend=:topleft)

        # Set the x and y axis labels
        xlabel!(p2, "Episode")
        ylabel!(p2, "Probability")
        # ================================================================================

        p = plot(p1, p2, layout=(2,1))
        savefig(
            p, 
            joinpath(path, "learningcurve.pdf")
            )
        return p
    end

    """
    This method saves the results of the given experiment.

    fpath: the path to the file where the results will be saved.
    rec: the record of the performance of the policy.
    rews: an array of the rewards obtained at each timestep and 
    stored in the bandit history.
    piflag: an array of booleans that specify whether, at each timestep, 
    the safe policy was used because the computed one was unsafe.
    save_res: a boolean that specifies whether the results should be saved
    in a CSV file.
    """
    function save_results(
        rec::SafetyPerfRecord, 
        rews, 
        tidxs, 
        piflag,
    )
        # unsafe 
        unsafe = zeros(length(rews))
        # notnsf = a mask that indicates whether, at the given timestep index, the candidate policy
        # was used (because considered safe)
        notnsf = zeros(length(rews))
        # For each starting and ending timestep in the tidxs array
        for (i, (ts, te)) in enumerate(tidxs)
            # If the computed new policy was used against the safe one (i.e., passed the safety test)
            if piflag[i]
                # Set to 1 the values of the notnsf array from the start to the end timestep
                notnsf[ts:te] .= 1
                # If the mean return of the safe policy would have been higher than the one of the computed policy
                if mean(rec.Jsafe[ts:te]) > mean(rec.Jpi[ts:te])
                    # Set to 1 the values of the unsafe array from the start to the end timestep
                    unsafe[ts:te] .= 1
                end
            end
        end
        # Compute the mean performance as the mean of the rewards over all the timesteps
        obsperf = mean(rews)
        # Compute the mean performance of the candidate policy as the mean of the Jpi values
        canperf = mean(rec.Jpi)
        # Compute the mean performance of the algorithm as the mean of the Jpi values if the policy was safe,
        # otherwise the mean of the Jsafe values
        algperf = mean(notnsf .* rec.Jpi .+ (1 .- notnsf) .* rec.Jsafe)
        # Compute the foundpct as the percentage over all the timesteps in which the candidate policy was used
        # against the safe one
        foundpct = mean(notnsf)
        # Compute the violation index, that is the percentage of timesteps in which the candidate policy 
        # was used against the safe one but had a lower return
        violation = mean(unsafe)
        # Compute the regret as the mean between the difference of the algorithm performance and the safe policy performance
        # TODO Wouldn't it be more intuitive to call it gain? Because it is > 0 when the actual algorithm performances outperform
        # the safe policy ones
        regret = mean(notnsf .* rec.Jpi .+ (1 .- notnsf) .* rec.Jsafe .- rec.Jsafe)
        # Return all the results in a single array
        sumres = [obsperf, canperf, algperf, regret, foundpct, violation]
        return sumres
    end
end

"""
Generate a random number from a log-uniform distribution between low and high
using the provided random number generator. The code calculates the 
log-probability of the generated number.

The number and the log-probability are returned.
"""
function logRand(low, high, rng)
    # Generate a random number from a uniform distribution between log(low) and log(high)
    random_value= rand(rng, Uniform(log(low), log(high)))
    # Then take the exponential of this number to get a number from a log-uniform distribution
    # (Then, log(X) has a uniform distribution between log(low) and log(high))
    X = exp(random_value)
    # PDF: 1 / ( X * log(high / low) )
    # log-PDF: -log(X) - log(log(high) - log(low))

    # Calculate the log-probability of X, which is calculated between log(low) and log(high)
    # TODO: log(high) - log(high) will always be 0, shouldn't it be log(high) - log(low)
    logp = -log(X) - log(log(high) - log(low)) # -log(log(high) - log(high) - log(X))

    # Return the generated number and its log-probability
    return X, logp
end

function logRand(low, high, rng)
    X = exp(rand(rng, Uniform(log(low), log(high))))
    logp = -log(log(high) - log(high) - log(X))
    return X, logp
end

"""
This method samples the hyperparameters for the optimization of the bandit safety problem.
These are the batch size, the entropy regularizer, the value used to calculate the 
number of new samples collected for each iteration for the policy optimization step, 
and the Fourier basis order used to model the bandit performances.

speed: the speed of non-stationarity of the bandit problem.
rng: the random number generator used to sample the hyperparameters.
"""
function sample_hyperparams(
    speed::Int,
    rng
)
    # Set the number of agent interactions (timesteps) to be performed
    # before the HICOPI step
    τ = rand(rng, [2,4,6,8])
    # Pick a random number between 0.00005 and 1.0 for
    # the entropy regularizer
    λ = logRand(0.00005, 1.0, rng)[1]
    # Pick a random number between 2 and 5
    opt_ratio = rand(rng) * 3 + 2
    # If the speed is zero (i.e., stationary environment)
    if speed == 0
        # Set the Fourier basis order to 0 (i.e., no change in the environment is needed,
        # therefore the only component is the constant one)
        fb_order = 0
    # Otherwise, set the Fourier basis order to a random number between 1 and 4
    else
        fb_order = rand(rng, 1:4)
    end
    # Return the sampled hyperparameters as a tuple in which tau and fborder are rounded to integers
    params = (round(Int, τ), λ, opt_ratio, round(Int, fb_order))
    return params
end

@everywhere function run_trials(
    seed,
    save_dir,
    n_trials,
    num_episodes,
    algname,
    speed,
    hyps,
    rng,
)
    # Create a name for the save file based on the algorithm name and seed
    save_name = "$(algname)_$(lpad(seed, 5, '0')).csv"

    # Join the save directory and save name to create the full save path
    save_path = joinpath(save_dir, save_name)

    # Open the file for writing the results
    file = open(save_path, "w")

    # Write the header line to the save file
    write(file, "tau,lambda,optratio,fborder,obsperf,canperf,algperf,regret,foundpct,violation\n")
    # Flush the file to ensure the header line is written
    flush(file)
    
    trial_results = []
    trial_hyperparams = []
    futures = []

    max_process_index = get_max_processes()
    proc_index = 2
    tot_processed = 0
    # Loop over the number of trials
    for t_index in 1:n_trials
        if proc_index == 1
            # raise an exception
            throw(ArgumentError("Too many processes"))
        end
        print("Started trial $t_index with hyperparameters $(hyps) on process $proc_index\n")
        # Run the optimization for the non-stationary bandit safety problem
        future = @spawnat proc_index optimize_nsdbandit_safety(
            num_episodes, 
            rng, 
            speed, 
            hyps, 
        )
        push!(futures, future)
        proc_index += 1
        
        if length(futures) == max_process_index - 1 || tot_processed == n_trials
            for future in futures
                res = fetch(future)
                # Remove the future from the list of futures
                futures = futures[2:end]
                tot_processed += 1
        
                # Add the results to the list of results
                push!(trial_results, res)
                # Add the hyperparameters to the list of hyperparameters
                push!(trial_hyperparams, hyps)
        
                # Create a string for the current experiment results, composed by:
                # - hyperparameters
                # - observed performance
                # - candidate performance
                # - algorithm performance
                # - regret (i.e., difference between algorithm and safe policy performance)
                # - found percentage (i.e., percentage of times the candidate policy was used)
                # - violation percentage (i.e., percentage of times the candidate policy was used but had a lower return)
                tmp_result_row = join([
                    hyps..., 
                    save_results(res[1], res[2], res[3], res[4])...
                    ], 
                    ','
                )
        
                # Write the results of the current experiment
                write(file, "$(tmp_result_row)\n")
        
                # Flush the file to ensure the result line is written
                flush(file)

                # If the t_index is not the last one, break and send a new job
                if t_index < n_trials
                    break
                end
                # Otherwise, consume all the futures
            end
        end
    end

    return trial_hyperparams, trial_results
end


"""
This method runs, for trials times, the sampling of the hyperparameters and the optimization of the
(non-stationary) bandit safety problem. The results are saved in a unique CSV file.

seed: the seed for the random number generator, to enable reproducibility of the results.

algname: the name of the algorithm that is used to optimize the bandit safety problem 
(either "stationary" or "nonstationary").

save_dir: the directory where the results will be saved.

trials: the number of trials to run.

speed: the speed of non-stationarity of the bandit problem.

num_episodes: the number of total timesteps that must be contained in each experiment.
"""
function runsweep(
    seed,
    save_dir, 
    trials, 
    speed, 
    num_episodes
)
    plots_path = joinpath(save_dir, "plots")
    # Build the "plots" folder, if not already existing
    if !isdir(plots_path)
        mkdir(plots_path)
    end

    # Initialize a random number generator with the provided seed
    rng = Random.MersenneTwister(seed)

        # Sample the hyperparameters with a Fourier basis order of 0
    hyps = sample_hyperparams(
        0,
        rng
    )
    """future_baseline = @spawnat 1 run_trials(
        seed,
        save_dir,
        trials,
        num_episodes,
        "BASELINE",
        speed,
        hyps,
        rng,
    )"""
    

    _, baseline_trial_results = run_trials(
        seed,
        save_dir,
        trials,
        num_episodes,
        "BASELINE",
        speed,
        hyps,
        rng,
        )

    # Replace the Fourier basis order with a random number in the SPIN algorithm experiment
    fourier_basis_order = round(Int, sample_hyperparams(speed, rng)[4])
    hyps = [
        hyps[1], 
        hyps[2], 
        hyps[3],
        # cast the following to Int64
        fourier_basis_order
    ]
    """future_spin = @spawnat 1 run_trials(
        seed,
        save_dir,
        trials,
        num_episodes,
        "SPIN",
        speed,
        hyps,
        rng,
    )

    _, baseline_trial_results = fetch(future_baseline)
    _, spin_trial_results = fetch(future_spin)"""

    _, spin_trial_results = run_trials(
        seed,
        save_dir,
        trials,
        num_episodes,
        "SPIN",
        speed,
        hyps,
        rng,
    )
    
    # Now combine the data of all the experiments of the BASELINE and SPIN algorithms
    baseline_combined_results = combine_trials(baseline_trial_results)
    spin_combined_results = combine_trials(spin_trial_results)
    # Get the pi_safe performance from the first trial of the first record (non stationary case)
    pi_safe_results = spin_trial_results[1][1].Jsafe

    df1 = DataFrame(
        t = baseline_combined_results[1],  
        mnsafe = baseline_combined_results[2][1], 
        stdsafe = baseline_combined_results[2][2], 
        mnfound = baseline_combined_results[3][1], 
        stdfound = baseline_combined_results[3][2], 
        mnJpi = baseline_combined_results[4][1], 
        stdJpi = baseline_combined_results[4][2], 
        mnJalg = baseline_combined_results[5][1], 
        stdJalg = baseline_combined_results[5][2], 
        baseline=pi_safe_results
    )
    # TODO shouldn't it be more clear to call this file baseline_learncurve.csv?
    CSV.write(
        joinpath(plots_path, "BASELINE_experiments_average.csv"),
        df1
    )
    
    df2 = DataFrame(
        t = spin_combined_results[1],  
        mnsafe = spin_combined_results[2][1], 
        stdsafe = spin_combined_results[2][2], 
        mnfound = spin_combined_results[3][1], 
        stdfound = spin_combined_results[3][2], 
        mnJpi = spin_combined_results[4][1], 
        stdJpi = spin_combined_results[4][2], 
        mnJalg = spin_combined_results[5][1], 
        stdJalg = spin_combined_results[5][2], 
        baseline=pi_safe_results
    )
    # TODO shouldn't it be more clear to call this file spin_learncurve.csv?
    CSV.write(
        joinpath(plots_path, "SPIN_experiments_average.csv"),
        df2
    )

    # Display the results of the BASELINE and SPIN algorithms
    # display(
    learning_curves(
        baseline_combined_results, 
        spin_combined_results, 
        pi_safe_results, 
        ["Baseline", "SPIN"], 
        "Nonstationary Recommender System",
        plots_path,
    )
    # )
end

function tmp(
    num_episodes, 
    speed
)
    # Build the "plots" folder, if not already existing
    if !isdir("plots")
        mkdir("plots")
    end
    # Set this random number generator seed for reproducibility
    rng = MersenneTwister(0)

    # Set the number of experiment trials to 30
    num_trials = 2 # 30 # TODO in the paper they specified that this is 10 for the RecSys experiment

    # NONSTATIONARY CASE
    # Set the hyperparameters: batch size, entropy regularizer, optimization ratio, and Fourier basis order
    hyps = sample_hyperparams(
        speed,
        rng
    ) # TODO before: (Int(4), 0.125, 3.0, Int(3))
    print(hyps)
    recs1 = [
        optimize_nsdbandit_safety(
            num_episodes, 
            rng, 
            speed, 
            hyps, 
            ) for _ in 1:num_trials
    ]
    
    # STATIONARY CASE
    # Set the hyperparameters: batch size, entropy regularizer, optimization ratio, and Fourier basis order
    hyps = sample_hyperparams(
        0,
        rng
    ) # TODO before: (Int(4), 0.125, 3.0, Int(0))
    # Replace in the tuple the Fourier basis order 
    # with 0 in the BASELINE algorithm experiment
    hyps = (hyps[1], hyps[2], hyps[3], 0)
    recs2 = [
        optimize_nsdbandit_safety(
            num_episodes, 
            rng, 
            speed, 
            hyps, 
            ) for _ in 1:num_trials
    ]

    r1 = combine_trials(recs1)
    r2 = combine_trials(recs2)
    
    # Get the baseline value from the first trial of the first record (non stationary case)
    baseline = recs1[1][1].Jsafe
    println(size.([r1[1],  r1[2][1], r1[2][2], r1[3][1], r1[3][2], r1[4][1], r1[4][2], r1[5][1], r1[5][2], baseline]))
    df1 = DataFrame(
        t = r1[1],  
        mnsafe = r1[2][1], 
        stdsafe = r1[2][2], 
        mnfound = r1[3][1], 
        stdfound = r1[3][2], 
        mnJpi = r1[4][1], 
        stdJpi = r1[4][2], 
        mnJalg = r1[5][1], 
        stdJalg = r1[5][2], 
        baseline=baseline
    )
    CSV.write("plots/nonstationary_learncurve.csv", df1)
    df2 = DataFrame(
        t = r2[1],  
        mnsafe = r2[2][1], 
        stdsafe = r2[2][2], 
        mnfound = r2[3][1], 
        stdfound = r2[3][2], 
        mnJpi = r2[4][1], 
        stdJpi = r2[4][2], 
        mnJalg = r2[5][1], 
        stdJalg = r2[5][2], 
        baseline=baseline
    )
    CSV.write("plots/stationary_learncurve.csv", df2)
    display(
        learning_curves(
            r2, 
            r1, 
            baseline, 
            ["Baseline", "SPIN"], 
            "Nonstationary Recommender System",
            "plots",
        )
    )
end

# TODO remove the following two functions
function isprime(n::Int)
    if n < 2
        return false
    end
    for i in 2:isqrt(n)
        if n % i == 0
            return false
        end
    end
    return true
end

function test()
    for i in 1:1000
        rng = MersenneTwister(0)
        # Draw a random number between 10 and 1000
        τ = rand(rng, 1000:20000)
        # Find all the primes between 1 and τ
        _ = [i for i in 1:τ if isprime(i)]
    end
end

# Main function to run the experiments
function main()
    # Create a settings object to hold the arguments
    s = ArgParseSettings()
    # Add arguments to the settings object
    @add_arg_table! s begin
        "--log-dir", "-l"
            help = "folder directory to store results"
            arg_type = String
            required = true
        "--id"
            help = "identifier for this experiment. Used in defining a random seed"
            arg_type = Int
            required = true
        "--seed", "-s"
            help = "random seed is seed + id"
            arg_type = Int
            required = true
        "--trials", "-t"
            help = "number of random trials to run"
            arg_type = Int
            default = 1
        "--speed"
            help = "speed of the environment [0,1,2,3]"
            arg_type = Int
            default = 1
        "--eps"
            help = "number of episodes to run"
            arg_type = Int
            default = 100

    end

    # Parse the command line arguments and set them into the s object
    parsed_args = parse_args(localARGS, s)

    # Flush the standard output to ensure that the previous prints are displayed
    flush(stdout)

    # Extract the save directory from the parsed arguments
    save_dir = parsed_args["log-dir"]

    # Extract the number of trials, id, seed, speed, and number of episodes from the parsed arguments
    trials = parsed_args["trials"]
    id = parsed_args["id"]
    seed = parsed_args["seed"]
    speed = parsed_args["speed"]
    num_episodes = parsed_args["eps"]

    # Create the directory path where to store the results 
    # (i.e., a subdirectory of the given directory, 
    # named "discretebandit_" followed by the speed of the
    # current environment)
    save_dir = joinpath(save_dir, "discretebandit_$speed", "experiment_$id")

    # Create the save directory and its parents if they do not exist
    mkpath(save_dir)

    # Print the id, seed, and their sum
    println(id, " ", seed, " ", id + seed)

    # Flush the standard output to ensure that the previous print is displayed
    flush(stdout)

    # Update the seed to be the sum of id and seed
    seed = id + seed

    @time begin
    # Run the sweep (i.e., a simulation of the experiment) both in the BASELINE
    # and SPIN cases in an environment whose (non)stationarity is determined by speed. 
    # For each of them, run the simulation for the given number
    # of trials ("num_episodes" each), saving the results in the given directory
    # Initialize the profiler with a larger buffer and/or larger delay
    # Buffer size = 10 million, delay = 0.01 seconds
    # Profile.init(n = 10^7, delay = 0.01)
    # ProfileView.@profview runsweep(
    runsweep(
        seed, 
        save_dir, 
        trials, 
        speed, 
        num_episodes
    )
    end
end

# If the newARGS variable is defined (i.e., the script is being run from the REPL),
if @isdefined(newARGS)
    # Set the localARGS variable to the newARGS variable
    localARGS = newARGS
else
    # Otherwise, set the localARGS variable to the ARGS variable;
    # NOTE: ARGS in Julia is a global variable that holds the command line arguments;
    localARGS = ARGS
end

# =======================================================================
# USE THE FOLLOWING to see the effect of the code execution over a 
# single / multiple processes
@everywhere function print1000strings(t, print_out=false)
    iter = 0
    for _ in 1:100000
        if print_out
            print("Multiplication in thread $t\n")
        end

        # Make a big memory allocation (0.2GB)
        _ = Array{Float64}(undef, 25_600_000)
        # Create two matrices of random numbers of dimension 1000x1000
        array1 = rand(1000, 10)
        array2 = rand(1000, 10)
        # Compute the scalar product of the two matrices
        _ = array1' * array2
        # println("Printing from thread $t: $scalar_product")
        # println("Printing from thread $t")
        # Append the scalar product to the list of scalar products
        iter += 1 
    end
    
    if !print_out
        return [iter, 0.4, "a", Dict("a" => iter)]
    end
end

function test(multi_process::Bool)
    cum_value = 0
    if multi_process
        futures = []
        for t in 1:max_process_index
            future = @spawnat t print1000strings(t)
            push!(futures, future)
        end

        for future in futures
            x = fetch(future)
            cum_value = Int(x[4]["a"]) + cum_value
        end
        print("Cumulative value: $cum_value\n")
    else
        for t in 1:max_process_index
            print1000strings(t)
        end
    end
end

# @time test(true)
# @time test(false)
# =======================================================================

# Call the main function to perform the experiment
main()