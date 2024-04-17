using Statistics
using EvaluationOfRLAlgs
using Random
using DataFrames
using CSV
using ArgParse
using Distributions


include("normalpolicy.jl")
include("softmaxpolicy.jl")
include("history.jl")
include("optimizers.jl")
include("offpolicy.jl")

include("environments.jl")
include("nonstationary_modeling.jl")
include("nonstationary_pi.jl")

struct SafetyPerfRecord{T1, T2} <: Any where {T1,T2}
    t::Array{T1,1}
    Jpi::Array{T2,1}
    Jsafe::Array{T2,1}

    function SafetyPerfRecord(::Type{T1}, ::Type{T2}) where {T1,T2}
        new{T1,T2}(Array{T1,1}(), Array{T2,1}(), Array{T2,1}())
    end
end

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

function record_perf!(rec::SafetyPerfRecord, eval_fn, t, πc, πsafe)
    push!(rec.t, t)
    push!(rec.Jpi, eval_fn(πc))
    push!(rec.Jsafe, eval_fn(πsafe))
end

function optimize_nsdbandit_safety(
    num_episodes, 
    rng, 
    speed, 
    hyperparams, 
    fpath, 
    save_res
)
    # Create a new BanditHistory object to store the history of the bandit problem
    D = BanditHistory(Float64, Int)
    # Set the payoffs for the arms of the bandit problem
    arm_payoffs = [1.0, 0.8, 0.6, 0.4, 0.2]
    # Set the array of frequencies for the arms of the bandit problem
    arm_freq = zeros(length(arm_payoffs))
    # If the non-stationarity speed is 0 (stationary environment)
    if speed == 0
        κ = 0 #(2*pi)/1500.0
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
    # Assign the frequency of the arms to the value stored in κ for each value in the array
    arm_freq .= κ
    # Create a new arm_k array, uninitialized, with the same length as the arm_freq array
    arm_k = similar(arm_freq)
    # Assign to each element of arm_k the value of pi/2
    arm_k .= pi/2
    # Multiply the 2*pi/5 to each element of the left-hand side array, then sum each element into the arm_k array
    arm_k .+= (2*pi/5) .* [0.0, 1.0, 2.0, 3.0, 4.0]
    # Create a new array of ones with the same length as the arm_payoffs array
    # that represents the Rademacher variable probabilities
    arm_sigma = ones(length(arm_payoffs))
    arm_sigma .*= 0.05

    # Create a new NonStationaryDiscreteBanditParams object with the arm_payoffs, arm_sigma, arm_freq, and arm_k arrays
    env = NonStationaryDiscreteBanditParams(
        arm_payoffs, 
        arm_sigma, 
        arm_freq, 
        arm_k, 
        [0.0]
    )

    # TODO this seems to be useless because the values are statically reassigned
    # θ = deepcopy(arm_payoffs) .* 0.4

    # Set the theta values (payoffs?) for each arm
    θ = [2.0, 1.5, 1.2, 1.0, 1.0]

    # Create a new policy pi to choose among the arm_payoffs
    π = StatelessSoftmaxPolicy(Float64, length(arm_payoffs))
    set_params!(π, θ)
    # Clone the built policy and assign it to πsafe
    πsafe = clone(π)

    # ===========================================================
    # Define a function with a short function definition syntax.
    env_fn(action, rng) = sample_reward!(env, action, rng)
    # ===========================================================
    
    sample_counter = [0]
    rec = SafetyPerfRecord(Int, Float64)
    eval_fn(π) = eval_policy(env, π)
    # ===========================================================
    function log_eval(action, rng, sample_counter, rec, eval_fn, π, πsafe)
        sample_counter[1] += 1
        record_perf!(rec, eval_fn, sample_counter[1], π, πsafe)
        return env_fn(action, rng)
    end
    # ===========================================================
    log_fn = (action, rng) -> log_eval(action, rng, sample_counter, rec, eval_fn, π, πsafe)
    # sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)
    sample_fn(D, π, N) = collect_data!(D, π, log_fn, N, rng)

    # Set the parameters for the optimization algorithm (Adam)
    oparams = AdamParams(get_params(π), 1e-2; β1=0.9, β2=0.999, ϵ=1e-5)

    τ, λ, opt_ratio, fborder = hyperparams

    nboot_train = 200 # num bootstraps
    nboot_test = 500
    # τ = 8 # num_steps to optimize for future performance and to collect data for
    δ = 0.05 # percentile lower bound to maximize future for (use 1-ϵ for upper bound)
    # aggf = mean # function to aggregate future performance over (e.g., mean over τ steps,) maximium and minimum are also useful
    # λ = 0.000005
    IS = PerDecisionImportanceSampling()
    # num_opt_iters = 40

    num_opt_iters = round(Int, τ*opt_ratio)
    warmup_steps = 20
    sm = SplitLastKKeepTest(0.25)  # Fraction of data samples to be used for training
    # Assign the normalization function to nt
    nt = normalize_time(D, τ)
    # Create a Fourier series with the fborder value
    fb = fourierseries(Float64, fborder)
    # Apply a Fourier series to the normalized time
    ϕ(t) = fb(nt(t))

    num_iters = num_episodes / τ
    # ϕ(t) = fb(t/1520)
    # ϕ(t) = [1.0]
    train_idxs = Array{Int, 1}()
    test_idxs = Array{Int, 1}()

    # opt_fun, safety_fun = build_nsbs(D, π, oparams, ϕ, τ; nboot_train=nboot_train, nboot_test=nboot_test, δ=δ, aggf=aggf, λ=λ, IS=IS, num_iters=num_opt_iters, rng=rng)
    opt_fun, safety_fun = build_nsbst(
        D, 
        π, 
        oparams, 
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

    res = save_results(fpath, rec, D.rewards, tidx, piflag, save_res)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    return res
    # return (rec, D.rewards, tidx, piflag)
end

function combine_trials(results)
    T = length(results)
    N = length(results[1][2])
    t = results[1][1].t
    unsafe = zeros((N, T))
    notnsf = zeros((N, T))
    Jpi = zeros((N, T))
    Jalg = zeros((N, T))
    for (k, (rec, rews, tidxs, piflag)) in enumerate(results)
        unf = zeros(N)
        nnsf = zeros(N)
        for (i, (ts, te)) in enumerate(tidxs)
            if piflag[i]
                nnsf[ts:te] .= 1
                if mean(rec.Jsafe[ts:te]) > mean(rec.Jpi[ts:te])
                    unf[ts:te] .= 1
                end
            end
        end
        @. unsafe[:, k] = unf
        @. notnsf[:, k] = nnsf
        @. Jpi[:, k] += rec.Jpi
        @. Jalg[:, k] += nnsf * rec.Jpi + (1 - nnsf) * rec.Jsafe
    end

    mnunsafe = vec(mean(unsafe, dims=2))
    stdunsafe = vec(std(unsafe, dims=2)) ./ √T
    mnfound = vec(mean(notnsf, dims=2))
    stdfound = vec(std(notnsf, dims=2)) ./ √T
    mnJpi = vec(mean(Jpi, dims=2))
    stdJpi = vec(std(Jpi, dims=2)) ./ √T
    mnJalg = vec(mean(Jalg, dims=2))
    stdJalg = vec(mean(Jalg, dims=2)) ./ √T
    # println(size(mnunsafe), " ", size(stdunsafe), " ", size(unsafe))
    return t, (mnunsafe, stdunsafe), (mnfound, stdfound), (mnJpi, stdJpi), (mnJalg, stdJalg)
end

function learning_curves(res1, res2, baseline, labels, title)
    p1 = plot(title=title)
    p2 = plot()

    # rgrad = cgrad([:crimson, :dodgerblue])
    # plot!(p1, rec.t, rews, linestyle=:dot, lc=rgrad, line_z=issafety, label="observed rewards")
    t1, safe1, found1, Jpi1, Jalg1 = res1
    plot!(p1, t1, baseline, lc=:black, label="π_safe")
    plot!(p1, t1, Jalg1[1], ribbon=Jalg1[2], lc=:crimson, fillalpha=0.3, label=labels[1])
    plot!(p1, t1, Jpi1[1], lc=:crimson, linestyle=:dot, label=nothing)

    t2, safe2, found2, Jpi2, Jalg2 = res2
    plot!(p1, t2, Jalg2[1], ribbon=Jalg2[2], lc=:dodgerblue, fillalpha=0.3, label=labels[2])
    plot!(p1, t2, Jpi2[1], lc=:dodgerblue, linestyle=:dot, label=nothing, legend=:topleft)

    xlabel!(p1, "Episode")
    ylabel!(p1, "Performance")

    plot!(p2, t1, found1[1], ribbon=found1[2], linestyle=:dash, lc=:crimson, fillalpha=0.3, label="Canditate Returned-Baseline")
    plot!(p2, t1, safe1[1], ribbon=safe1[2], linestyle=:solid, lc=:crimson, fillalpha=0.3,  label="Unsafe Policy-Baseline")
    plot!(p2, t2, found2[1], ribbon=found2[2], linestyle=:dash, lc=:dodgerblue, fillalpha=0.3,  label="Canditate Returned-SPIN")
    plot!(p2, t2, safe2[1], ribbon=safe2[2], linestyle=:solid, lc=:dodgerblue, fillalpha=0.3,  label="Unsafe Policy-SPIN", legend=:topleft)

    xlabel!(p2, "Episode")
    ylabel!(p2, "Probability")
    p = plot(p1, p2, layout=(2,1))
    savefig(p, "learningcurve.pdf")
    return p
end

function save_results(fpath, rec::SafetyPerfRecord, rews, tidxs, piflag, save_res=true)
    unsafe = zeros(length(rews))
    notnsf = zeros(length(rews))
    for (i, (ts, te)) in enumerate(tidxs)
        if piflag[i]
            notnsf[ts:te] .= 1
            if mean(rec.Jsafe[ts:te]) > mean(rec.Jpi[ts:te])
                unsafe[ts:te] .= 1
            end
        end
    end
    df = DataFrame(t = rec.t, rews=rews, Jpi = rec.Jpi, found=notnsf, unsafe=unsafe)
    if save_res
        CSV.write(fpath, df)
    end
    obsperf = mean(rews)
    canperf = mean(rec.Jpi)
    algperf = mean(notnsf .* rec.Jpi .+ (1 .- notnsf) .* rec.Jsafe)
    foundpct = mean(notnsf)
    violation = mean(unsafe)
    regret = mean(notnsf .* rec.Jpi .+ (1 .- notnsf) .* rec.Jsafe .- rec.Jsafe)
    sumres = [obsperf, canperf, algperf, regret, foundpct, violation]
    return sumres
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
    # 

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

function sample_ns_hyperparams(rng)
    # TODO what are these parameters?
    # Pick one of the values from the list [2,4,6,8]
    τ = rand(rng, [2,4,6,8])
    # Pick a random number between 0.00005 and 1.0
    λ = logRand(0.00005, 1.0, rng)[1]
    # Pick a random number between 2 and 5
    opt_ratio = rand(rng)*3 + 2
    # Pick a random number between 1 and 4 (inclusive)
    fborder = rand(rng, 1:4)
    # Return the sampled hyperparameters as a tuple in which tau and fborder are rounded to integers
    params = (round(Int, τ), λ, opt_ratio, round(Int, fborder))
    return params
end

function sample_stationary_hyperparams(rng)
    # TODO what are these parameters?
    # Pick one of the values from the list [2,4,6,8]
    τ = rand(rng, [2,4,6,8])
    # Pick a random number between 0.00005 and 1.0
    λ = logRand(0.00005, 1.0, rng)[1]
    # Pick a random number between 2 and 5
    opt_ratio = rand(rng)*3 + 2
    fborder = 0
    # Return the sampled hyperparameters as a tuple in which tau and fborder are rounded to integers
    params = (round(Int, τ), λ, opt_ratio, round(Int, fborder))
    return params
end


# Function to run a sweep of trials for a given algorithm
function runsweep(seed, algname, save_dir, trials, speed, num_episodes)
    # Initialize a random number generator with the provided seed
    rng = Random.MersenneTwister(seed)

    # Create a name for the save file based on the algorithm name and seed
    save_name = "$(algname)_$(lpad(seed, 5, '0')).csv"

    # Join the save directory and save name to create the full save path
    save_path = joinpath(save_dir, save_name)

    # Open the save file for writing
    file = open(save_path, "w")
    
    # Write the header line to the save file
    write(file, "tau,lambda,optratio,fborder,obsperf,canperf,algperf,regret,foundpct,violation\n")
    # Flush the file to ensure the header line is written
    flush(file)

    # Loop over the number of trials
    for trial in 1:trials
        # If the algorithm name is "stationary", sample stationary hyperparameters
        # Otherwise, sample non-stationary hyperparameters
        if algname == "stationary"
            hyps = sample_stationary_hyperparams(rng)
        else
            hyps = sample_ns_hyperparams(rng)
        end

        # Run the optimization for the non-stationary bandit safety problem
        res = optimize_nsdbandit_safety(num_episodes, rng, speed, hyps, "nopath.csv", false)

        # Join the hyperparameters and results into a single string
        result = join([hyps..., res...], ',')

        # Write the result string to the save file
        write(file, "$(result)\n")

        # Flush the file to ensure the result line is written
        flush(file)
    end
end

function tmp(num_episodes, speed)
    # hyps = (Int(2), 0.06125, 3.0, Int(3))
    num_trials = 30
    hyps = (Int(4), 0.125, 3.0, Int(3))
    rng = MersenneTwister(0)
    recs1 = [optimize_nsdbandit_safety(num_episodes, rng, speed, hyps, "nopath.csv", false) for i in 1:num_trials]

    hyps = (Int(4), 0.125, 3.0, Int(0))
    # rng = MersenneTwister(0)
    recs2 = [optimize_nsdbandit_safety(num_episodes, rng, speed, hyps, "nopath.csv", false) for in in 1:num_trials]

    r1 = combine_trials(recs1)
    r2 = combine_trials(recs2)
    baseline = recs1[1][1].Jsafe
    # println(size.([r1[1],  r1[2][1], r1[2][2], r1[3][1], r1[3][2], r1[4][1], r1[4][2], r1[5][1], r1[5][2], baseline]))
    df1 = DataFrame(t = r1[1],  mnsafe = r1[2][1], stdsafe = r1[2][2], mnfound = r1[3][1], stdfound = r1[3][2], mnJpi = r1[4][1], stdJpi = r1[4][2], mnJalg = r1[5][1], stdJalg = r1[5][2], baseline=baseline)
    CSV.write("nonstationary_learncurve.csv", df1)
    df2 = DataFrame(t = r2[1],  mnsafe = r2[2][1], stdsafe = r2[2][2], mnfound = r2[3][1], stdfound = r2[3][2], mnJpi = r2[4][1], stdJpi = r2[4][2], mnJalg = r2[5][1], stdJalg = r2[5][2], baseline=baseline)
    CSV.write("stationary_learncurve.csv", df2)
    display(learning_curves(r2, r1, baseline, ["Baseline", "SPIN"], "Nonstationary Recommender System"))
end

# Main function to run the experiments
function main()
    # Create a settings object to hold the arguments
    s = ArgParseSettings()
    # Add arguments to the settings object
    @add_arg_table! s begin
        "--alg-name", "-a"
            help = "name of the algorithm to run"
            arg_type = String
            required = true
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
    # (ARGS in Julia is a global variable that holds the command line arguments)
    parsed_args = parse_args(ARGS, s)

    # Extract the algorithm name from the parsed arguments
    aname = parsed_args["alg-name"]

    # Print the algorithm name and the log directory
    println(aname)
    println(parsed_args["log-dir"])

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

    # Join the save directory with a subdirectory named "discretebandit_" followed by the speed
    save_dir = joinpath(save_dir, "discretebandit_$speed")

    # Create the save directory and its parents if they do not exist
    mkpath(save_dir)

    # Print the id, seed, and their sum
    println(id, " ", seed, " ", id + seed)

    # Flush the standard output to ensure that the previous print is displayed
    flush(stdout)

    # Update the seed to be the sum of id and seed
    seed = id + seed

    # Run the sweep with the specified parameters
    runsweep(seed, aname, save_dir, trials, speed, num_episodes)

end

main()
