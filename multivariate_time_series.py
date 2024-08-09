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
    return forecast[:, 0]  # Forecast only for Series_1

def fit_bayesian_var(data):
    # Convert DataFrame to NumPy array
    data_array = data.values

    # Define the number of time series and lags
    n_series = data_array.shape[1]
    n_lags = 1  # Number of lags, update if necessary

    with pm.Model() as model:
        # Define priors for AR coefficients and noise
        phi = pm.Normal('phi', mu=0, sigma=1, shape=(n_series, n_series))  # AR coefficients
        sigma = pm.HalfNormal('sigma', sigma=1)  # Standard deviation of the noise

        # Likelihood function
        mu = pm.math.dot(data_array[:-1, :], phi)  # mu for the likelihood
        obs = pm.Normal('obs', mu=mu[:, 0], sigma=sigma, observed=data_array[1:, 0])  # Target is Series_1
        
        # Sample from the posterior
        trace = pm.sample(1000, return_inferencedata=False, progressbar=True, tune=1000)

        # Forecast for Series_1
        forecast_samples = np.empty((10,))  # 10 time steps
        last_observed = data_array[-1, :]  # Last observed values for Series_1 to Series_5
        phi_mean = trace['phi'].mean(axis=0)  # Mean of the posterior samples
        sigma_mean = trace['sigma'].mean()  # Mean of the posterior samples for sigma
        
        for t in range(10):
            mu_t = np.dot(last_observed, phi_mean)  # Predict Series_1
            forecast_samples[t] = mu_t[0]  # Store the forecast value
            last_observed = np.hstack([forecast_samples[t], last_observed[:-1]])  # Update for next time step
        
        # Calculate the 95% confidence intervals
        forecast_trace = np.empty((1000, 10))  # 1000 posterior samples, 10 forecast steps
        for i in range(1000):
            forecast_i = np.empty((10,))
            last_observed_i = data_array[-1, :]  # Last observed values
            for t in range(10):
                mu_t_i = np.dot(last_observed_i, trace['phi'][i])  # Predict
                forecast_i[t] = mu_t_i[0]  # Store the forecast value
                last_observed_i = np.hstack([forecast_i[t], last_observed_i[:-1]])  # Update for next time step
            forecast_trace[i] = forecast_i
        
        lower_bound = np.percentile(forecast_trace, 2.5, axis=0)
        upper_bound = np.percentile(forecast_trace, 97.5, axis=0)

        return forecast_samples, lower_bound, upper_bound

def combine_forecasts(bayesian_forecast, var_forecast):
    combined_forecast = (bayesian_forecast + var_forecast) / 2
    return combined_forecast

def plot_forecasts(original_data, var_forecast, bayesian_forecast, combined_forecast, lower_bound, upper_bound):
    plt.figure(figsize=(14, 7))
    plt.plot(original_data.index, original_data['Series_1'], label='Original Series 1', color='blue')
    plt.plot(np.arange(len(original_data), len(original_data) + len(var_forecast)), var_forecast, label='Standard VAR Forecast', color='red')
    plt.plot(np.arange(len(original_data), len(original_data) + len(bayesian_forecast)), bayesian_forecast, label='Bayesian VAR Forecast', color='green')
    plt.plot(np.arange(len(original_data), len(original_data) + len(combined_forecast)), combined_forecast, label='Combined Forecast', color='purple')
    plt.fill_between(np.arange(len(original_data), len(original_data) + len(combined_forecast)), lower_bound, upper_bound, color='purple', alpha=0.2, label='95% Confidence Interval')
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

    print("Standard VAR Forecast for Series_1:")
    print(var_forecast)

    # Fit and forecast using PyMC model
    bayesian_forecast, lower_bound, upper_bound = fit_bayesian_var(data)

    print("Bayesian VAR Forecast (PyMC) for Series_1:")
    print(bayesian_forecast)

    # Combine forecasts
    combined_forecast = combine_forecasts(bayesian_forecast, var_forecast)

    print("Combined Forecast for Series_1:")
    print(combined_forecast)

    # Plot results
    plot_forecasts(data, var_forecast, bayesian_forecast, combined_forecast, lower_bound, upper_bound)

if __name__ == '__main__':
    main()
