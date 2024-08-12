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
    # Forecasting
    forecast = model_fitted.forecast(model_fitted.endog[-model_fitted.k_ar:], steps=steps)
    
    # Extract covariance matrix
    sigma_u = model_fitted.sigma_u
    if sigma_u.ndim == 3:
        sigma_u = np.mean(sigma_u, axis=0)
    
    stderr = np.sqrt(np.diagonal(sigma_u))
    # Repeat stderr for each forecast step
    stderr = np.tile(stderr, (steps, 1))
    conf_ints = 1.96 * stderr
    
    return forecast, conf_ints

def fit_bayesian_var(data):
    with pm.Model() as model:
        # Define priors for AR coefficients and noise
        phi = pm.Normal('phi', mu=0, sigma=1, shape=(5, 5))
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Likelihood function
        mu = pm.math.dot(data[:-1], phi)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data[1:])
        
        # Sample from the posterior
        trace = pm.sample(1000, return_inferencedata=False, progressbar=True, tune=1000)
        
        # Compute Bayesian forecasts and confidence intervals
        forecast_samples = np.random.randn(10, 5)  # Replace with actual Bayesian forecasts
        forecast_mean = np.mean(forecast_samples, axis=0)
        forecast_std = np.std(forecast_samples, axis=0)
        conf_ints = 1.96 * forecast_std
        # Ensure shapes are correct
        forecast_mean = np.expand_dims(forecast_mean, axis=0)
        conf_ints = np.expand_dims(conf_ints, axis=0)
        return forecast_mean, conf_ints

def combine_forecasts(bayesian_forecast, var_forecast):
    combined_forecast = (bayesian_forecast + var_forecast) / 2
    return combined_forecast

def plot_forecasts(original_data, var_forecast, var_conf_ints, bayesian_forecast, bayesian_conf_ints, combined_forecast):
    plt.figure(figsize=(14, 7))
    
    # Original Series 1
    plt.plot(original_data.index, original_data['Series_1'], label='Original Series 1', color='blue')
    
    # Standard VAR Forecast and Confidence Interval
    forecast_index = np.arange(len(original_data), len(original_data) + len(var_forecast))
    plt.plot(forecast_index, var_forecast[:, 0], label='Standard VAR Forecast', color='red')
    plt.fill_between(forecast_index, 
                     var_forecast[:, 0] - var_conf_ints[:, 0], 
                     var_forecast[:, 0] + var_conf_ints[:, 0], 
                     color='red', alpha=0.2, label='VAR 95% CI')
    
    # Bayesian VAR Forecast and Confidence Interval
    forecast_index = np.arange(len(original_data), len(original_data) + len(bayesian_forecast))
    plt.plot(forecast_index, bayesian_forecast[:, 0], label='Bayesian VAR Forecast', color='green')
    plt.fill_between(forecast_index, 
                     bayesian_forecast[:, 0] - bayesian_conf_ints[:, 0], 
                     bayesian_forecast[:, 0] + bayesian_conf_ints[:, 0], 
                     color='green', alpha=0.2, label='Bayesian VAR 95% CI')
    
    # Combined Forecast
    forecast_index = np.arange(len(original_data), len(original_data) + len(combined_forecast))
    plt.plot(forecast_index, combined_forecast[:, 0], label='Combined Forecast', color='purple')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Original Data and Forecasts with 95% Confidence Intervals')
    plt.legend()
    plt.show()

def main():
    # Generate data
    data = generate_synthetic_data()

    # Print the synthetic data
    print("Synthetic Data:")
    print(data.head())

    # Save the synthetic data to CSV
    data.to_csv('synthetic_data.csv', index=False)

    # Fit and forecast using standard VAR model
    var_model_fitted = fit_standard_var(data, lags=1)
    var_forecast, var_conf_ints = forecast_standard_var(var_model_fitted, steps=10)

    print("Standard VAR Forecast:")
    print(var_forecast)
    print("Standard VAR Confidence Intervals:")
    print(var_conf_ints)

    # Fit and forecast using PyMC model
    bayesian_forecast, bayesian_conf_ints = fit_bayesian_var(data)

    print("Bayesian VAR Forecast (PyMC):")
    print(bayesian_forecast)
    print("Bayesian VAR Confidence Intervals:")
    print(bayesian_conf_ints)

    # Combine forecasts
    combined_forecast = combine_forecasts(bayesian_forecast, var_forecast)

    print("Combined Forecast:")
    print(combined_forecast)

    # Plot results
    plot_forecasts(data, var_forecast, var_conf_ints, bayesian_forecast, bayesian_conf_ints, combined_forecast)

if __name__ == '__main__':
    main()
