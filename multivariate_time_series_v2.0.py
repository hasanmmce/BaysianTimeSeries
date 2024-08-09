import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
from statsmodels.tsa.api import VAR

def generate_synthetic_data(n=100):
    np.random.seed(42)
    data = np.random.randn(n, 5)  # Five time series
    df = pd.DataFrame(data, columns=['Series_1', 'Series_2', 'Series_3', 'Series_4', 'Series_5'])
    return df

def fit_standard_var(data, lags=1):
    model = VAR(data)
    model_fitted = model.fit(lags)
    return model_fitted

def forecast_standard_var(model_fitted, steps=10):
    forecast = model_fitted.forecast(model_fitted.endog, steps=steps)
    return forecast[:, 0]  # Forecast for Series_1

def fit_bayesian_var(data):
    data_array = data.values
    n_series = data_array.shape[1]
    forecast_steps = 10

    with pm.Model() as model:
        # Priors for AR coefficients and noise
        phi = pm.Normal('phi', mu=0, sigma=1, shape=(n_series, n_series))
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Likelihood function
        mu = pm.math.dot(data_array[:-1], phi)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data_array[1:])
        
        # Sample from the posterior
        trace = pm.sample(1000, return_inferencedata=False, progressbar=True, tune=1000)
        
        # Forecast for all series
        forecast_samples = np.empty((forecast_steps, n_series))
        last_observed = data_array[-1]
        phi_mean = trace['phi'].mean(axis=0)
        sigma_mean = trace['sigma'].mean()
        
        for t in range(forecast_steps):
            mu_t = np.dot(last_observed, phi_mean)
            forecast_samples[t] = mu_t
            last_observed = np.hstack([forecast_samples[t], last_observed[:-n_series]])
        
        # Calculate 95% confidence intervals
        forecast_trace = np.empty((1000, forecast_steps, n_series))
        for i in range(1000):
            forecast_i = np.empty((forecast_steps, n_series))
            last_observed_i = data_array[-1]
            for t in range(forecast_steps):
                mu_t_i = np.dot(last_observed_i, trace['phi'][i])
                forecast_i[t] = mu_t_i
                last_observed_i = np.hstack([forecast_i[t], last_observed_i[:-n_series]])
            forecast_trace[i] = forecast_i
        
        lower_bound = np.percentile(forecast_trace, 2.5, axis=0)
        upper_bound = np.percentile(forecast_trace, 97.5, axis=0)

        return forecast_samples[:, 0], lower_bound[:, 0], upper_bound[:, 0]

def combine_forecasts(bayesian_forecast, var_forecast):
    # Averaging forecasts for Series_1
    combined_forecast = (bayesian_forecast + var_forecast) / 2
    return combined_forecast

def plot_forecasts(original_data, var_forecast, bayesian_forecast, combined_forecast, bayesian_lower, bayesian_upper):
    plt.figure(figsize=(14, 7))
    plt.plot(original_data.index, original_data['Series_1'], label='Original Series 1', color='blue')
    plt.plot(np.arange(len(original_data), len(original_data) + len(var_forecast)), var_forecast, label='Standard VAR Forecast', color='red')
    plt.plot(np.arange(len(original_data), len(original_data) + len(bayesian_forecast)), bayesian_forecast, label='Bayesian VAR Forecast', color='green')
    plt.plot(np.arange(len(original_data), len(original_data) + len(combined_forecast)), combined_forecast, label='Combined Forecast', color='purple')

    plt.fill_between(np.arange(len(original_data), len(original_data) + len(bayesian_forecast)), bayesian_lower, bayesian_upper, color='green', alpha=0.3, label='Bayesian 95% CI')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Original Data and Forecasts')
    plt.legend()
    plt.show()

def main():
    # Generate data
    data = generate_synthetic_data()

    # Print the synthetic data
    print("Synthetic Data:")
    print(data.head())

    # Fit and forecast using standard VAR model
    var_model_fitted = fit_standard_var(data, lags=1)
    var_forecast = forecast_standard_var(var_model_fitted, steps=10)

    print("Standard VAR Forecast for Series_1:")
    print(var_forecast)

    # Fit and forecast using PyMC model
    bayesian_forecast, bayesian_lower, bayesian_upper = fit_bayesian_var(data)

    print("Bayesian VAR Forecast (PyMC) for Series_1:")
    print(bayesian_forecast)
    print("Bayesian 95% CI Lower Bound:")
    print(bayesian_lower)
    print("Bayesian 95% CI Upper Bound:")
    print(bayesian_upper)

    # Combine forecasts
    combined_forecast = combine_forecasts(bayesian_forecast, var_forecast)

    print("Combined Forecast for Series_1:")
    print(combined_forecast)

    # Plot results
    plot_forecasts(data, var_forecast, bayesian_forecast, combined_forecast, bayesian_lower, bayesian_upper)

if __name__ == '__main__':
    main()
