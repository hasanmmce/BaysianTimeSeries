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
    return forecast

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
        
        # Placeholder for Bayesian forecasts
        bayesian_forecast = np.random.randn(10, 5)  # Replace with actual Bayesian forecasts
        return bayesian_forecast

def combine_forecasts(bayesian_forecast, var_forecast):
    # Example: Averaging forecasts for Series_1
    combined_forecast = (bayesian_forecast[:, 0] + var_forecast[:, 0]) / 2
    return combined_forecast

def plot_forecasts(original_data, var_forecast, bayesian_forecast, combined_forecast):
    plt.figure(figsize=(14, 7))
    plt.plot(original_data.index, original_data['Series_1'], label='Original Series 1', color='blue')
    plt.plot(np.arange(len(original_data), len(original_data) + len(var_forecast)), var_forecast[:, 0], label='Standard VAR Forecast', color='red')
    plt.plot(np.arange(len(original_data), len(original_data) + len(bayesian_forecast)), bayesian_forecast[:, 0], label='Bayesian VAR Forecast', color='green')
    plt.plot(np.arange(len(original_data), len(original_data) + len(combined_forecast)), combined_forecast, label='Combined Forecast', color='purple')
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

    # Save the synthetic data to CSV
    data.to_csv('synthetic_data.csv', index=False)

    # Fit and forecast using standard VAR model
    var_model_fitted = fit_standard_var(data, lags=1)
    var_forecast = forecast_standard_var(var_model_fitted, steps=10)

    print("Standard VAR Forecast:")
    print(var_forecast)

    # Fit and forecast using PyMC model
    bayesian_forecast = fit_bayesian_var(data)

    print("Bayesian VAR Forecast (PyMC):")
    print(bayesian_forecast)

    # Combine forecasts
    combined_forecast = combine_forecasts(bayesian_forecast, var_forecast)

    print("Combined Forecast:")
    print(combined_forecast)

    # Plot results
    plot_forecasts(data, var_forecast, bayesian_forecast, combined_forecast)

if __name__ == '__main__':
    main()
