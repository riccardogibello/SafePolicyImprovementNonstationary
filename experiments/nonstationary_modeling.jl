using Zygote: @adjoint
using Statistics
using LinearAlgebra

"""
    get_coefs(Φ, ϕτ)

This method returns least squares coefficients for making predictions
at training timesteps (i.e., W, related to the Φ matrix) 
and future timesteps (i.e., ϕ, related to the ϕτ matrix).

Φ: the feature matrix for the training timesteps, calculated through the basis function ϕ.
ϕτ: the feature matrix for the future timesteps, calculated through the basis function ϕ.

# TODO not very clear explanation
This function is a helper function to be used for predicting
through linear regression future points in a time series 
and in wild bootstrap.
"""
function get_coefs(Φ, ϕτ)
    # Calculate the pseudo-inverse of the matrix Φ using the Moore-Penrose inverse;
    H = pinv(Φ' * Φ) * Φ'
    # Calculate the least squares coefficients (hat matrix) for making predictions of Y hat
    # in the training timesteps
    W = Φ * H
    # Calculate the least squares coefficients (hat matrix) for making predictions of the 
    # performances in the future timesteps τ
    ϕ = ϕτ * H
    return W, ϕ
end

"""
    get_preds_and_residual(Y, W, ϕ)

This method returns the baseline future performance predictions for the time points τ,
calculated through the fitted past returns Y at the training timesteps.
The vector of residuals between the observed and the fitted returns is also returned.

Y: the observed returns at the training timesteps.
W: the least squares coefficients for making predictions of Ŷ in the training timesteps.
ϕ: the least squares coefficients for making predictions of the performances in the future timesteps τ.
"""
function get_preds_and_residual(Y, W, ϕ)
    # Calculate the fitted Ŷ return values for the training timesteps 
    # (i.e., using the W matrix and the observed Y values).
    # Ŷ = Φ * H * Y = W * Y
    Ŷ = W * Y
    # Predict, based on the past performances Ŷ, the future performances
    # in the time points τ
    y = ϕ * Ŷ
    # Calculate the residuals between the returns in the 
    # training timesteps (i.e., Y) and the fitted returns of the 
    # training timesteps (i.e., Ŷ)
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
This method returns least squares coefficients for making predictions
at training timesteps (i.e., W, related to the Φ matrix) and the 
future timesteps (i.e., ϕ, related to the ϕτ matrix) and the TODO.

Φ: the feature matrix for the training timesteps, calculated through the basis function ϕ.
ϕτ: the feature matrix for the future timesteps, calculated through the basis function ϕ.

# TODO not very clear explanation
This function is a helper function to be used for predicting
through linear regression future points in a time series 
and in wild bootstrap.
"""
function get_coefst(Φ, ϕτ)
    # Calculate the pseudo-inverse of the matrix Φ using the Moore-Penrose inverse;
    H = pinv(Φ' * Φ) * Φ'
    # Calculate the least squares coefficients (hat matrix) for making predictions of Y hat
    # in the training timesteps
    W = Φ * H
    # Calculate the least squares coefficients (hat matrix) for making predictions of the 
    # performances in the future timesteps τ
    ϕ = ϕτ * H
    # Calculate the difference between the identity matrix and the hat matrix.
    # This is a projection matrix transforms the target values into the 
    # residuals of the linear regression model.
    # The residuals are the differences between the target values and the predicted values, 
    # and they are orthogonal to the predicted values (i.e., the component that cannot
    # be explained by a linear combination of the features).
    C = I - W
    return W, ϕ, C
end

"""

ϕ: the result of the product between the ϕτ and H.
"""
function get_preds_and_residual_t(Y, W, ϕ, C)
    # Project the observed values Y onto the column space of Φ, to obtain
    # the predicted values Ŷ of the performances
    # Ŷ = Φ * H * Y = W * Y
    Ŷ = W * Y
    # Calculate the predicted performances by transforming the predicted values 
    # Ŷ with the matrix ϕ. Later the mean is computed on this vector.
    y = ϕ * Ŷ
    # Calculate the residuals of the predicted values Ŷ by projecting Ŷ onto 
    # the orthogonal complement of the column space of Φ. 
    # These are the component of Ŷ that cannot be explained by a linear 
    # combination of the features (i.e., if there is random noise,
    # non-linear relationships, important features ignored in the model).
    # This can be used for assessing the fit of the model and for 
    # bootstrapping procedures, where the residuals are resampled to generate 
    # new datasets for model validation.
    x = C * Ŷ
    # Get the residuals between the observed values Y and the predicted values Ŷ
    ξ = Y .- Ŷ
    # Being ξ a vector of independent noises, the Diagonal(ξ.^2) is a diagonal matrix
    # representing the co-variance matrix of the mean-zero and heteroscedastic noises ξ.
    # Then, Σ corresponds to the Vf matrix of the paper because:
    # Σ =   ϕτ * H * Diagonal(ξ.^2) * H' * ϕτ' = 
    #       ϕτ * pinv(Φ' * Φ) * Φ' * Diagonal(ξ.^2) * Φ * pinv(Φ' * Φ) * ϕτ'
    Σ = ϕ * Diagonal(ξ.^2) * ϕ'
    # Get the mean variance of the residuals between the training predictions and 
    # observations
    v = mean(Σ)
    
    return y, v, x, ξ
end

"""
x: the residuals of the fitted returns in the training samples that cannot 
be explained by the model (e.g., random noise, non-linear relationships,
important features ignored).
C: the projection matrix that transforms the target predicted values into 
the residuals (i.e., the portion that cannot be explained by the model).
ξ: the residuals between the observed and the fitted returns in the training samples.
ϕ: the least squares coefficients (hat matrix) for making predictions of 
the performances in the future timesteps τ.
"""
function wildbst_eval(x, C, ξ, ϕ, σ)
    # Calculate the perturbed residuals ξ̂ = ξ .* σ (actually ξ̂* of the pseudocode)
    ξ̂ = ξ .* σ
    ξ̂2 = (x .+ C * ξ̂ ).^2
    # Predict the mean future error of the predicted performances with respect to the actual 
    # performances
    Δŷ = mean(ϕ * ξ̂ )
    # Compute the co-variance matrix of the estimator of the error over the future predictions
    Σ = ϕ * Diagonal(ξ̂2) * ϕ'
    # Calculate the mean variance of the estimator of the error over the future predictions
    v̂ = mean(Σ)

    # Return the standardized mean error over the future predicted performances,
    # calculated as the ratio between:
    # 1) the mean error over the future predicted performances;
    # 2) the standard deviation of the estimator of the error over 
    # the future predictions (line 14 of the pseudocode);
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

"""
This method performs a bootstrap procedure to estimate the actual mean
future predicted performance, taking into account the possible variability
of the predictions.

y: the predicted future performances.
v: the value of the mean variance of the residuals that cannot be explained
by the fitting model on the training samples, wrt the observed ones.
x: the residuals of the fitted returns in the training samples that cannot 
be explained by the model (e.g., random noise, non-linear relationships,
important features ignored).
ξ: the residuals between the observed and the fitted returns in the training samples.
ϕ: the least squares coefficients (hat matrix) for making predictions of 
the performances in the future timesteps τ.
C: the projection matrix that transforms the target predicted values into 
the residuals (i.e., the portion that cannot be explained by the model).
δ: the confidence level.
num_boot: the number of bootstrap samples.
rng: the random number generator.
"""
function wildbst_CI(y, v, x, ξ, ϕ, C, δ, num_boot,rng)
    # Generate a number of num_boot bootstrap samples of the standardized mean error
    # over future predicted performances
    bsamples = [
        wildbst_eval(
            x,
            C,
            ξ,
            ϕ,
            # Set the list of random perturbations (+-1) to the residuals ξ
            sign.(randn(rng, length(ξ)))
        )
        for _ in 1:num_boot
    ]
    # Calculate the mean of the predicted future performances
    p = mean(y)
    # Calculate the actual future performance prediction, by subtracting the
    # (1.0 - δ)-th quantile of the standardized mean error over the future
    # predicted performances. Therefore, this will represent the upper
    # bound of a 1.0 - δ confidence interval.
    return p - quantile(sort(bsamples), 1.0 - δ, sorted=true) * √v # bst_CIs(p, v, bsamples, δ)
end

# defines the adjoint function for sorting so Zygote can automatically differentiate the sort function.
@adjoint function sort(x)
     p = sortperm(x)
     x[p], x̄ -> (x̄[invperm(p)],)
end

"""
This method, using basis function ϕ, creates the feature matrices of the 
observed training timesteps (i.e., x vector), and future time points τ 
(shifted of the history L time points).

ϕ: the fourier transform function (see "fourierseries" method), that accepts a scalar 
(i.e. element of x vector) and returns a vector of |C| length, containing the 
element-wise cosine of the product between C and x.
x: the observed training timesteps.
τ: the future timesteps, shifted of the history L time points.
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
