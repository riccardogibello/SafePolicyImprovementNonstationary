# =======================================================================
# TODO these lines are needed only when run without the shell script
# Get the parent directory path
@eval __DIRPARENT__ = dirname(dirname(@__FILE__))
# Get the current OS
@eval os = Sys.iswindows() ? "windows" : "unix"
# Concatenate the parent directory path with the python directory
if os == "windows"
    python_directory = joinpath(__DIRPARENT__, ".venv\\Scripts\\python.exe")
else
    python_directory = joinpath(__DIRPARENT__, ".venv/bin/python3")
end
println("Python directory path: ", python_directory)
ENV["PYTHON"] = python_directory
# =======================================================================

using Statistics
using EvaluationOfRLAlgs
using Random
using DataFrames
using CSV
using ArgParse
using Distributions

include("../normalpolicy.jl")
include("../softmaxpolicy.jl")
include("../history.jl")
include("../optimizers.jl")
include("../offpolicy.jl")

include("../environments.jl")
include("glucose_env.jl")
using .glucose_env
include("../nonstationary_modeling.jl")
include("../nonstationary_pi.jl")

include("optimization.jl")
using .optimization:optimize_nsglucose_safety, optimize_disc_nsglucose_safety, optimize_nscbandit_safety

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

function plot_results(rews, tidxs, piflag, title)
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
        end
        scatter!(p1, ts:te, rews[ts:te], markercolor=color, markersize=1, markerstrokewidth=0, label=nothing)
        issafety[ts:te] .= piflag[i]
    end
    rgrad = cgrad([:crimson, :dodgerblue])
    plot!(p1, rec.t, rews, linestyle=:dot, lc=rgrad, line_z=issafety, label="observed rewards")
    # plot!(p1, rec.t, rec.Jsafe, lc=:crimson, label="J(π_safe)")
    # plot!(p1, rec.t, rec.Jpi, lc=:dodgerblue, label="J(π_c)", legend=:bottomleft)
    # plot!(p2, )
    xlabel!(p1, "Episode")
    ylabel!(p1, "Return")

    plot!(p2, rec.t, notnsf, label="Canditate Returned")
    # plot!(p2, rec.t, unsafe, label="Unsafe Policy", legend=:bottomright)
    xlabel!(p2, "Episode")
    ylabel!(p2, "Probability")
    p = plot(p1, p2, layout=(2,1))
    savefig(p, "myplot.pdf")
    return p
end

"""
This method loads and returns the matrices and means of the evaluation data for the glucose environment.

speed: Int - The speed of the glucose environment, that is prefixed to any file name.
"""
function load_eval(speed)
    mats = [
        # Convert each CSV file to a matrix
        # NOTE: the |> operator is used to pipe the output of the CSV.File function to the Tables.matrix function
        CSV.File(
            joinpath(@__DIR__, "glucose_eval_data", "speed$(speed)_allact$(i).csv"), 
            header=false
        ) |> Tables.matrix for i in 1:5
    ]
    # For each matrix, calculate the mean of each column (i.e., by specifying the dims=1 argument to the mean function)
    means = [
        mean(mat, dims=1) 
        for mat in mats
    ]
    return mats, means
end

function nsglucose_safety(
    num_episodes, 
    rng, 
    speed,
    is_discrete,
)
    if is_discrete
        # Create an history object for keeping all the actions taken, 
        # their log-probabilities, and the rewards received
        D = BanditHistory(Float64, Int)
        # Instantiate the glucose environment with the given non-stationarity speed
        # and a value for the seed to be given to the FDA simulator
        env::NSDiscreteGlucoseSim = NSDiscreteGlucoseSim(speed, abs(rand(rng, UInt32)))

        # Set the initial probabilities for each action
        p = [0.5, 0.125, 0.125, 0.125, 0.125]
        # Normalize the probabilities to avoid overflow/underflow
        # when dealing with probabilities
        θ = log.(p) .- mean(log.(p))
        θ .= exp.(θ) / sum(exp.(θ))

        # Assign the θ vector to the π policy (and initialize the probabilities
        # to the softmax of the θ vector)
        π:: StatelessSoftmaxPolicy = StatelessSoftmaxPolicy(Float64, length(p))
    else
        # Create an history object for keeping all the actions taken, 
        # their log-probabilities, and the rewards received
        D = BanditHistory(Float64, Array{Float64,1})
        # Instantiate the glucose environment with the given non-stationarity speed
        # and a value for the seed to be given to the FDA simulator
        env::NSGlucoseSim = NSGlucoseSim(speed, abs(rand(rng, UInt32)))

        # Set the initial parameters for the policy
        θ = [20.0, 30.0, 2.0, 2.0]

        # Instantiate an initial policy, which is a normal policy with two actions
        # and a standard deviation of 2.0 for each action
        π::StatelessNormalPolicy = StatelessNormalPolicy(Float64, 2, 2.0, true)
    
    set_params!(π, θ)
    πsafe = clone(π)

    # Set the function to draw the rewards given an action from the
    # FDA simulator (the ignored parameter is due to the shared method)
    env_fn(action, _) = sample_reward!(env, action)
    sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)

    # Set the number of episodes to be run in the simulation before the 
    # policy iteration updates
    warmup_steps = 20
    # Set the total number of timesteps to the sum of the number of episodes
    # and the number of warmup steps
    N = num_episodes + warmup_steps

    # Perform the sampling of the returns by using the safe policy for the
    # total number of timesteps
    sample_fn(D, πsafe, N)
    # Set the unique interval of interaction from 1 to N
    tidx = [(1,N)]
    # Set the flag that indicates that the safe policy was always used
    piflag = [false]

    return (rec, D.rewards, tidx, piflag)
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

# function save_results(fpath, rews, tidxs, piflag, save_res=true)
#     notnsf = zeros(length(rews))
#     for (i, (ts, te)) in enumerate(tidxs)
#         if piflag[i]
#             notnsf[ts:te] .= 1
#         end
#     end
#     df = DataFrame(t = 1:length(rews), rews=rews, found=notnsf)
#     if save_res
#         CSV.write(fpath, df)
#     end
#     obsperf = mean(rews)
#     foundpct = mean(notnsf)

#     sumres = [obsperf, foundpct]
#     return sumres
# end

function logRand(low, high, rng)
    X = exp(rand(rng, Uniform(log(low), log(high))))
    logp = -log(log(high) - log(high) - log(X))
    return X, logp
end

function sample_hyperparams(
    rng, 
    old,
    alg_name,
)
    # Set the number of agent interactions (timesteps) to be performed
    # before the HICOPI step
    τ = rand(rng, [2,4,6,8])
    # Pick a random number between 0.01 and 1.0 for
    # the entropy regularizer
    λ = logRand(0.01, 1.0, rng)[1]
    # Pick a random number between 2 and 5 for the coefficient to be used
    # to calculate the total number of optimization iterations;
    # # optimization iterations
    opt_ratio = rand(rng) * 3 + 2
    
    # If the name of the algorithm is the BASELINE
    if alg_name == :baseline
        # Set the order of the Fourier basis expansion to 0
        fborder = 0
    else
        # Pick a random number between 1 and 4 for the order of the Fourier basis expansion
        fborder = rand(rng, 1:4)
    end
    
    # Return the sampled hyperparameters as a tuple in which tau and fborder are rounded to integers
    params = (round(Int, τ), λ, opt_ratio, round(Int, fborder), old)
    return params
end


function runsweep(seed, algname, save_dir, trials, speed, num_episodes)
    # Instantiate the random number generator with the given seed
    rng = Random.MersenneTwister(seed)

    save_name = "$(algname)_$(lpad(seed, 5, '0')).csv"


    save_path = joinpath(save_dir, save_name)

    open(save_path, "w") do f
        write(f, "tau,lambda,optratio,fborder,old_ent,obsperf,canperf,algperf,regret,foundpct,violation\n")
        flush(f)

        # For each trial to be executed in the given experiment
        for trial in 1:trials
            rets_name = "returns_$(algname)_$(lpad(seed, 5, '0'))_$(lpad(trial, 5, '0')).csv"
            ret_path = joinpath(save_dir, rets_name)
            if algname == "stationary"
                hyps = sample_hyperparams(
                    rng, 
                    false,
                    :baseline,
                    )
            elseif algname == "stationary-old"
                hyps = sample_hyperparams(
                    rng, 
                    true,
                    :baseline,
                    )
            elseif algname == "nonstationary-old"
                hyps = sample_hyperparams(
                    rng, 
                    true,
                    :spin,
                    )
            else
                hyps = sample_hyperparams(
                    rng, 
                    false,
                    :spin,
                    )
            end
            res = optimize_nsglucose_safety(
                num_episodes, 
                rng, 
                speed, 
                hyps, 
                ret_path, 
                true
                )
            res = optimize_disc_nsglucose_safety(
                num_episodes, 
                rng, 
                speed, 
                hyps, 
                ret_path, 
                true, 
                seed
            )

            result = join([hyps..., res...], ',')
            write(f, "$(result)\n")
            flush(f)
            # println("$trial \t $(result)")
            # flush(stdout)
        end
    end

end

"""
This method runs the experiment using always the safe policy, using the continuous and the
discrete FDA simulators.
"""
function runsafety(
    seed, 
    save_dir, 
    trials, 
    speed, 
    num_episodes
)
    rng = Random.MersenneTwister(seed)

    # For each of the trial that must be run for the current experiment
    for trial in 1:trials
        # Create the name and the path of the file for storing the mean values of the
        # current trial
        mean_perf_name = "safety_$(lpad(seed, 5, '0')).csv"
        mean_perf_path = joinpath(save_dir, mean_perf_name)
        # Create a file (override if existing) for storing the mean performances of the trial
        open(mean_perf_path, "w") do fmean
            write(fmean, "tau,lambda,optratio,fborder,obsperf,canperf,algperf,regret,foundpct,violation\n")
            flush(fmean)
        end
        fmean = open(mean_perf_path, "a")
        mean_discrete_name = "safety_discrete_$(lpad(seed, 5, '0')).csv"
        mean_discrete_path = joinpath(save_dir, mean_discrete_name)
        # Create a file (override if existing) for storing the mean performances of the discrete trial
        open(mean_discrete_path, "w") do fmean_discrete
            write(fmean_discrete, "tau,lambda,optratio,fborder,obsperf,canperf,algperf,regret,foundpct,violation\n")
            flush(fmean_discrete)
        end
        fmean_discrete = open(mean_discrete_path, "a")
        rets_name = "returns_safety_$(lpad(seed, 5, '0'))_$(lpad(trial, 5, '0')).csv"
        rets_discrete_name = "returns_discrete_safety_$(lpad(seed, 5, '0'))_$(lpad(trial, 5, '0')).csv"
        ret_path = joinpath(save_dir, rets_name)
        ret_discrete_path = joinpath(save_dir, rets_discrete_name)
        
        # 1) Run the trial by using the continuous FDA simulator
        res = nsglucose_safety(
            num_episodes, 
            rng, 
            speed,
            false
        )
        result = join([
            hyps...,
            # Save the timestep results and compute the mean performances of the trial
            save_results(ret_path, res[1], res[2], res[3], res[4], save_res)
            ], 
            ','
        )
        write(fmean, "$(result)\n")
        flush(fmean)
        
        # 2) Run the trial by using the discrete FDA simulator
        res = nsglucose_safety(
            num_episodes, 
            rng, 
            speed,
            true
        )
        result = join([
            hyps...,
            # Save the timestep results and compute the mean performances of the trial
            save_results(ret_discrete_path, res[1], res[2], res[3], res[4], save_res)
            ], 
            ','
        )
        write(fmean_discrete, "$(result)\n")
        flush(fmean_discrete)
    end

end

function tmp2(num_episodes, speed)
    # hyps = (Int(2), 0.06125, 3.0, Int(3))
    num_trials = 2
    hyps = (Int(4), 0.0001, 3.0, Int(3), false)
    rng = MersenneTwister(0)
    # optimize_nsglucose_safety(num_episodes, rng, speed, hyps, "nopath.csv", false)
    recs1 = [optimize_nscbandit_safety(num_episodes, rng, speed, hyps, "nopath.csv", false) for i in 1:num_trials]

    hyps = (Int(4), 0.0001, 3.0, Int(3), true)
    # rng = MersenneTwister(0)
    recs2 = [optimize_nscbandit_safety(num_episodes, rng, speed, hyps, "nopath.csv", false) for in in 1:num_trials]

    r1 = combine_trials(recs1)
    r2 = combine_trials(recs2)
    baseline = recs1[1][1].Jsafe
    # println(size.([r1[1],  r1[2][1], r1[2][2], r1[3][1], r1[3][2], r1[4][1], r1[4][2], r1[5][1], r1[5][2], baseline]))
    df1 = DataFrame(t = r1[1],  mnsafe = r1[2][1], stdsafe = r1[2][2], mnfound = r1[3][1], stdfound = r1[3][2], mnJpi = r1[4][1], stdJpi = r1[4][2], mnJalg = r1[5][1], stdJalg = r1[5][2], baseline=baseline)
    # CSV.write("nonstationary_learncurve.csv", df1)
    df2 = DataFrame(t = r2[1],  mnsafe = r2[2][1], stdsafe = r2[2][2], mnfound = r2[3][1], stdfound = r2[3][2], mnJpi = r2[4][1], stdJpi = r2[4][2], mnJalg = r2[5][1], stdJalg = r2[5][2], baseline=baseline)
    # CSV.write("stationary_learncurve.csv", df2)
    display(learning_curves(r2, r1, baseline, ["SPIN", "SPIN-OLD"], "Nonstationary Recommender System"))
    # display(learning_curves(r2, r1, baseline, ["Baseline", "SPIN"], "Nonstationary Recommender System"))
end

function main()
    s = ArgParseSettings()
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

    parsed_args = parse_args(ARGS, s)
    aname = parsed_args["alg-name"]
    println(aname)
    println(parsed_args["log-dir"])
    flush(stdout)
    save_dir = parsed_args["log-dir"]

    trials = parsed_args["trials"]
    id = parsed_args["id"]
    seed = parsed_args["seed"]
    speed = parsed_args["speed"]
    num_episodes = parsed_args["eps"]

    # Create the directory for the experiment
    save_dir = joinpath(save_dir, "dglucose_$speed/experiment_$id")
    mkpath(save_dir)


    println(id, " ", seed, " ", id + seed)
    flush(stdout)
    # Create the seed
    seed = id + seed
    if aname == "safety"
        runsafety(
            seed, 
            save_dir, 
            trials, 
            speed, 
            num_episodes
        )
    else
        runsweep(
            seed, 
            aname, 
            save_dir, 
            trials, 
            speed, 
            num_episodes
        )
    end

end

main()
