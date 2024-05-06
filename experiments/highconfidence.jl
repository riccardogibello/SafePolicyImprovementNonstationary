using EvaluationOfRLAlgs

"""
    safety_lastk_split(D, K)

Split the data D up into train and test sets by
randomly choosing half of the last K samples to be
in the test set and the rest in the train. This is
useful when data in D has been used to make policy
improvements in the past, but the K samples are new.

# Examples
```julia-repl
#D is a list of N trajectories
#π is a policy to collect data
K = 10# is the number of episodes to collect
collect_data!(D, π, K)
train_idxs, test_idxs = safety_lastk_split(D, K)
D[train_idxs]; returns the list of 1:N plus half of the last K samples
D[test_idxs]; returns half the last K samples
```
"""
function safety_lastk_split(D, K)
    N = length(D)
    idxs = randperm(K)
    k = floor(Int, K/2)
    train = vcat(1:N, N .+ idxs[1:k])
    test = N .+ idxs[k+1:end]
    return train, test
end


"""
    HICOPI_safety_test

This function performs a high confidence safety test of policy π
in reference to a policy πsafe. The evaluation function f computes
the confidence interval for a policy and level δ. The either
returns a symbol indicating if π :safe or :uncertain. :uncertain
means that it cannot be gauranteed that π is better the πsafe.
"""
function HICOPI_safety_test(f, π, πsafe, δ)
    pilow = f(π, δ/2.0, :left)
    safehigh = f(πsafe, δ/2.0, :right)
    # println(round(pilow-safehigh, digits=3), "\t", round.(π.probs, digits=2), "\t", round.(get_params(π), digits=3))#, " ", get_params(πsafe))
    # println(round(pilow-safehigh, digits=3), "\t", round.(get_params(π), digits=3))#, " ", get_params(πsafe))
    # println("π low, πsafe high: $pilow, $safehigh")
    if pilow > safehigh
        return :safe
    else
        return :uncertain
    end
end

"""
    HICOPI(D, sample_fn!, optimize_fn, confidence_test, π, πsafe, τ, δ)

TODO update this description
This is a high confidence off-policy policy improvement function that
performs a single iteration of collecting τ data samples with policy π,
finding a canditate policy, πc, and then checking to see if πc is
better than the safe policy, πsafe, with confidence δ. If πc is better
return πc otherwise return :NSF no solution found.

This function is generic and takes as input:
D, previous data, which is possibly empty
sample_fn!, a function to sample new data points using policy π
optimize_fn, a function to find a new policy on data D
confidence_test, a function that computes a high confidence upper or lower bound on a policies performance
π, a policy to collect data with and initialize the optimization search with
πsafe, a policy that is consider a safe baseline that can always be trusted
τ, number of samples to collect data for. τ could represent the number of episodes or draws from a bandit.
δ, confidence level to use to ensure that safe policies or :NSF is return with probability at least 1-δ.
"""
function HICOPI_step!(
    oparams, 
    π, 
    D, 
    train_idxs, 
    test_idxs, 
    optimize_fn!, 
    confidence_bound, 
    πsafe, 
    δ
)
    # Optimize the π policy on the training data and using the optimization parameters
    optimize_fn!(oparams, π, D, train_idxs)

    # If there is no test-data, then ignore the safety test_idxs
    # This can only happen when split ratio of train:test = 100:0
    if length(test_idxs) < 1
        # :safe is a symbol; symbols in Julia are a type of scalar 
        # that are often used to represent names of variables, functions, etc.
        # Return that the kept policy must be the safe one
        result = :safe
    else
        # Get the result of the safety check
        result = HICOPI_safety_test(
            (π, δ, tail) -> confidence_bound(D, test_idxs, π, δ, tail), 
            π, 
            πsafe, 
            δ
        )
    end

    # If the result of the safety check is uncertain
    if result == :uncertain
        # Return that no solution was found
        return :NSF
    else
        # Otherwise, return the current policy
        return π
    end
end

function HICOPI!(
    oparams, 
    π, 
    D::BanditHistory, 
    train_idxs, 
    test_idxs, 
    sample_fn!, 
    optimize_fn!, 
    confidence_bound, 
    πsafe, 
    τ, 
    δ, 
    split_method, 
    num_iterations, 
    warmup_steps
)
    # Set the behavior policy to be the safe policy
    πbehavior = πsafe

    timeidxs = Array{Tuple{Int,Int},1}()
    # Set an array to keep track of whether the policy improvement condition was met (with safety)
    picflag = Array{Bool, 1}()
    # Set the flag to false, indicating that the current behaviour policy is not the safe one
    using_pic = false
    # Add to the time index array the first set of indices, which are the warmup_steps
    push!(timeidxs, (1,warmup_steps))
    # Add to the picflag array the value false, indicating that the policy improvement condition was not met (false)
    push!(picflag, using_pic)
    # Collect data for the warmup steps
    collect_and_split!(
        D, 
        train_idxs, 
        test_idxs, 
        πbehavior, 
        warmup_steps, 
        sample_fn!, 
        split_method
    )

    # For each value between 1 and the number of iterations ("eps" parameter)
    for i in 1:num_iterations
        # Take the current length of the bandit history
        n = length(D)
        # Add the tuple (n+1, n+τ) to the time index array, which represents the amount
        # of steps to be performed with the current policy
        push!(timeidxs, (n+1, n+τ))
        # Set the value to indicate whether the currently used policy is the safe one
        push!(picflag, using_pic)
        # Collect data for the next τ steps by using the current behavior policy
        # and store the data in the bandit history
        collect_and_split!(
            D, 
            train_idxs, 
            test_idxs, 
            πbehavior, 
            τ, 
            sample_fn!, 
            split_method
        )
        # Perform the high confidence off-policy policy improvement step, which will
        # return the new policy if it is safe, or the safe policy if the new policy is not safe
        result = HICOPI_step!(
            oparams, 
            π, 
            D, 
            train_idxs, 
            test_idxs, 
            optimize_fn!, 
            confidence_bound, 
            πsafe, 
            δ
        )
        # If the current safety test does not allow to set the current policy pi as safe
        if result == :NSF
            # Keep the safe policy as the behavior policy
            πbehavior = πsafe
            println("iteration $i π is not safe")
            using_pic = false
        else
            # Otherwise, set the current policy as the behavior policy
            πbehavior = π
            println("iteration $i π is safe")
            using_pic = true
        end
    end
    return timeidxs, picflag
end

abstract type AbstractSplitMethod end

#unbaised method for splitting
struct SplitLastK{T} <: AbstractSplitMethod where {T}
    p::T
end

# biased method for splitting
struct SplitLastKKeepTest{T} <: AbstractSplitMethod where {T}
    p::T
end


"""
This method collects samples and splits them into train and test sets.
All the data are stored in train_idxs and test_idxs.
"""
function collect_and_split!(
    D::BanditHistory, 
    train_idxs, 
    test_idxs, 
    π, 
    N, 
    sample_fn!, 
    split_method::SplitLastK
)
    # Get the length of the bandit history
    L = length(D)
    # Use the given function to get N samples given the policy π and store the results in D
    sample_fn!(D, π, N)
    # Create a random permutation of the integer numbers from 1 to N
    idxs = randperm(N)
    # Find the number of samples, based on the given percentage, that must be used for training
    k = floor(Int, split_method.p*N)
    # Clear the current arrays of train and test indexes
    empty!(train_idxs)
    empty!(test_idxs)
    # Add again to the train indexes all the indexes from 1 to L
    append!(train_idxs, 1:L)
    append!(train_idxs, L .+ idxs[1:k])
    append!(test_idxs, L .+ idxs[k+1:end])
end

"""
This method collects samples and splits them into train and test sets.
All the data are stored in train_idxs and test_idxs.
"""
function collect_and_split!(
    D::BanditHistory, 
    train_idxs, 
    test_idxs, 
    π, 
    N, 
    sample_fn!, 
    split_method::SplitLastKKeepTest
)
    # Get the length of the bandit history
    L = length(D)
    # Use the given function to get N samples given the policy π and store the results in D
    sample_fn!(D, π, N)
    # Create a random permutation of the integer numbers from 1 to N
    idxs = randperm(N)
    # Find the number of samples, based on the given percentage, that must be used for training
    k = floor(Int, split_method.p*N)
    # Add to the proper array all the indexes to be used for training and testing 
    # (each value is offset by L, that are the current number of samples in the bandit history)
    append!(train_idxs, L .+ idxs[1:k])
    append!(test_idxs, L .+ idxs[k+1:end])
end
