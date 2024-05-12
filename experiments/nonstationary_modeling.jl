using Zygote: @adjoint
using Statistics
using LinearAlgebra

"""
    get_coefs(Φ, ϕτ)

returns least squares coefficients for making predictions
at observed points Φ and points of interest ϕτ.

This function is a helper function to be used for predicting
future points in a time series and for in wild bootstrap.
"""
function get_coefs(Φ, ϕτ)
    # Calculate the pseudo-inverse of the matrix Φ using the Moore-Penrose inverse;
    H = pinv(Φ' * Φ) * Φ'
    # Calculate the least squares coefficients for making predictions of Y hat
    W = Φ * H
    # Calculate the product between the transformed tau and the H, that is used 
    # in predicting the performances
    ϕ = ϕτ * H
    return W, ϕ
end

"""
    get_preds_and_residual(Y, W, ϕ)

returns the baseline predictions using observed labels Y at points ϕ
and along with the vector of residuals Y - Ŷ.
W and ϕ should be the output of get_coefs.
"""
function get_preds_and_residual(Y, W, ϕ)
    # Calculate the predicted Y values starting from the W and the observed Y
    # Ŷ = Φ * H * Y = W * Y
    Ŷ = W * Y
    # print(size(Ŷ))
    # print(size(ϕ))
    # TODO MSG is this the prediction of the performances calculation? (rho hat)
    y = ϕ * Ŷ
    # print(size(y))
    # Residuals between the observed values of Y and the predicted values of Y
    # TODO MSG is this the ξ hat in the pseudocode?
    ξ = Y .- Ŷ
    return y, ξ
end

"""
    wildbs_eval(y, ξ, ϕ, σ, f)

evaluates the prediction of a linear timeseries model using
noise generated for the wild bootstrap.

y is the prediction at points ϕ using original labels Y
ξ are the residual between the labels Y and prediction Ŷ
ϕ are the points (coefficients) at which the predictions y are made
σ is the noise sample used to modify the residual, e.g., vector of {-1, 1}
f is the function to aggregate the result for all points in ϕ, e.g., sum, mean, maximum, minimum
"""
function wildbs_eval(y, ξ, ϕ, σ, f)
    # Perturb the values of the y hat (pseudosamples of the performances)
    # with the noise
    # ξ .* σ is the ξ*
    r = f(y .+ ϕ * (ξ .* σ))
    # Returns the average (float) of the predicted performances
    return r
end

"""
    wildbs_CI(y, ξ, ϕ, δ, num_boot, aggf)

    ϕ is the result of the product between the Fourier transform of the time points and H

    num_boot is the number of train samples used for bootstrapping the confidence interval

computes the δ percentiles bootstrap using the wild bootstrap method with num_boot bootstrap samples
for original predictions y, with residuals ξ, at features ϕ, aggregated by aggf.
"""
function wildbs_CI(
    y, 
    ξ, 
    ϕ, 
    δ, 
    num_boot, 
    aggf, 
    rng
)
    # Generate the wild-bootstrap samples (num_boot) from averaging the 
    # values of the perturbed performances
    bootstrapping_samples = [wildbs_eval(
        y, 
        ξ,
        # This is the result from the product of ϕτ and H
        ϕ,
        # Create, using the passed random number generator, a vector of the same length as ξ,
        # containing random values of -1 and 1 (i.e. the sign of the random number)
        sign.(randn(rng, length(ξ))), 
        aggf
    ) for i in 1:num_boot]
    
    return quantile(
        sort(bootstrapping_samples),
        # δ is set to 0.05 at the beginning
        δ,
        # sorted indicates that the given vector is already sorted
        sorted=true
    )
end

"""
This method computes the coefficients of a non-stationary model.
"""
function get_coefst(Φ, ϕτ)
    H = pinv(Φ' * Φ) * Φ'
    W = Φ * H
    ϕ = ϕτ * H
    # TODO what is this C matrix?
    C = I - W
    return W, ϕ, C
end

"""

ϕ: the result of the product between the ϕτ and H.
"""
function get_preds_and_residual_t(Y, W, ϕ, C)
    # Calculate the predicted Y values starting from the W and the observed Y
    # Ŷ = Φ * H * Y = W * Y
    Ŷ = W * Y
    # Calculate the predicted performances (later on the mean is computed)
    y = ϕ * Ŷ
    x = C * Ŷ
    # Get the residuals between the observed and the predicted performances 
    ξ = Y .- Ŷ
    # Being ξ a vector of independent noises, the Diagonal(ξ.^2) is a diagonal matrix
    # representing the co-variance matrix of the mean-zero and heteroscedastic noises ξ.
    # Then, Σ corresponds to the Vf matrix of the paper because:
    # Σ =   ϕτ * H * Diagonal(ξ.^2) * H' * ϕτ' = 
    #       ϕτ * pinv(Φ' * Φ) * Φ' * Diagonal(ξ.^2) * Φ * pinv(Φ' * Φ) * ϕτ'
    Σ = ϕ * Diagonal(ξ.^2) * ϕ'
    # Get the mean variance of the errors
    v = mean(Σ)
    
    return y, v, x, ξ
end

function wildbst_eval(x, C, ξ, ϕ, σ)
    ξ̂ = ξ .* σ
    ξ̂2 = (x .+ C * ξ̂).^2
    Δŷ = mean(ϕ * ξ̂)
    Σ = ϕ * Diagonal(ξ̂2) * ϕ'
    v̂ = mean(Σ)

    return Δŷ / √v̂
end


# TODO this function is not used in the code
function bst_CIs(p, v, bsamples, δ)
    return p - quantile(sort(bsamples), 1.0-δ, sorted=true) * √v
end

# TODO this function is not used in the code
function bst_CIs(p, v, bsamples, δ::Array{T,1}) where {T}
    return p .- quantile(sort(bsamples), 1.0 .- δ, sorted=true) .* √v
end

function wildbst_CI(y, v, x, ξ, ϕ, C, δ, num_boot,rng)
    bsamples = [wildbst_eval(x, C, ξ, ϕ, sign.(randn(rng, length(ξ)))) for i in 1:num_boot]
    # Calculate the mean of the predicted performances (line 5 of Algorithm 1 in the pseudocode)
    p = mean(y)
    return p - quantile(sort(bsamples), 1.0-δ, sorted=true) * √v #bst_CIs(p, v, bsamples, δ)
end

# defines the adjoint function for sorting so Zygote can automatically differentiate the sort function.

@adjoint function sort(x)
     p = sortperm(x)
     x[p], x̄ -> (x̄[invperm(p)],)
end

"""
This method, using basis function ϕ, creates the feature matrices of the 
observed test time points x, and future time points τ (shifted of the previous 
L time points).

ϕ: the fourier transform function (see "fourierseries" method), that accepts a scalar 
(i.e. element of x vector) and returns a vector of |C| length, containing the 
element-wise cosine of the product between C and x.
x: the observed test time points indices.
τ: the future time points indices.
"""
function create_features(ϕ, x, τ)
    # Create the omega matrix, by applying the function ϕ to each scalar in x and concatenating
    # the results horizontally; 
    # the ' is the transposition operator, so each feature vector is a row;
    Φ = hcat(ϕ.(x)...)'
    # Create the Fourier transform of the tau values (time points), in the same way as above
    ϕτ = hcat(ϕ.(τ)...)'
    return Φ, ϕτ
end

"""
This function accepts a type T and a Fourier basis order.

    It returns a function that accepts a scalar x and returns a vector of |C| length, 
    containing the element-wise cosine of the product between C and x.
"""
function fourierseries(::Type{T}, order::Int) where {T}
    # create an array of incremental values (of type T) from 0 to order 
    # and multiply each of them by pi
    C = collect(T, 0:order) .* π
    # Return a function that accepts a scalar x and returns a vector of |C| length, 
    # containing the element-wise cosine of the product between C and x
    return x -> @. cos(C * x)
end

"""
Given a BanditHistory object and a number of steps for future performance, returns a function
    that takes as input x (time instant) and returns its normalized value (dividing by the sum of the
    length of the bandit history and the number of steps for future performance).

"""
function normalize_time(D, τ)
    # Return an anonymous function that normalizes the x value by dividing it by the length of D plus τ
    return x -> begin x / (length(D) + τ) end
end
