import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the daily revinue data
df = pd.read_csv('monthly_cost_index_data.csv', index_col='date', parse_dates=True, dayfirst=False)
df.head()

# Plot the daily revenue data
df['cost'].plot(title='Daily cost', figsize=(12, 6))
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
# Seasonal decomposition
result = seasonal_decompose(df['cost'], model='multivarite', period=12).plot()
plt.show()

from pmdarima import auto_arima
# split the data into train and test sets
train = df.iloc[:-12]
test = df.iloc[-12:]

#ARIMA model 
arima_model = auto_arima(train['cost'], seasonal=False, m=12, trace=True)

print(arima_model.summary())

# predicting with the ARIMA model
test_predictions = arima_model.predict(n_periods=12)
print(test_predictions)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# Evaluate the model
mse = mean_squared_error(test['cost'], test_predictions)
mae = mean_absolute_error(test['cost'], test_predictions)
mape = mean_absolute_percentage_error(test['cost'], test_predictions)
print(f'MSE: {mse}', f'MAE: {mae}', f'MAPE: {mape}', sep='\n')

#SARIMA model
sarima_model = auto_arima(train['cost'], seasonal=True, m=12, trace=True)

# predicting with the SARIMA model
test_predictions = sarima_model.predict(n_periods=12)
print(test_predictions)

# Evaluate the model
mse = mean_squared_error(test['cost'], test_predictions)
mae = mean_absolute_error(test['cost'], test_predictions)
mape = mean_absolute_percentage_error(test['cost'], test_predictions)
print(f'MSE: {mse}', f'MAE: {mae}', f'MAPE: {mape}', sep='\n')

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['cost'], label='Train')
plt.plot(test.index, test['cost'], label='Test')
plt.plot(test.index, test_predictions, label='Predictions')
plt.legend()
plt.show()


#SARIMAX model
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Fit the SARIMAX model
X_train = train.iloc[:, 1:]
X_test = test.iloc[:, 1:]

sarimax_model = SARIMAX(train['cost'], seasonal=True, m=12, exog=X_train)
sarimax_model_fit = sarimax_model.fit()
print(sarimax_model_fit.summary())

# Predict with the SARIMAX model
test_predictions = sarimax_model_fit.predict(start=len(train), end=len(train)+len(test)-1, exog=X_test)
print(test_predictions)

# Evaluate the model
mse = mean_squared_error(test['cost'], test_predictions)
mae = mean_absolute_error(test['cost'], test_predictions)
mape = mean_absolute_percentage_error(test['cost'], test_predictions)
print(f'MSE: {mse}', f'MAE: {mae}', f'MAPE: {mape}', sep='\n')

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['cost'], label='Train')
plt.plot(test.index, test['cost'], label='Test')
plt.plot(test.index, test_predictions, label='Predictions')
plt.legend()
plt.show()












# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Set a random seed for reproducibility
# np.random.seed(42)

# # Generate a date range for 5 years (monthly data from January 2019 to December 2023)
# dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='M')

# # Generate random data
# cost = np.random.uniform(50, 150, size=len(dates))
# par2 = np.random.uniform(10, 50, size=len(dates))
# par3 = np.random.uniform(5, 30, size=len(dates))

# # Create a DataFrame
# df = pd.DataFrame({
#     'date': dates,
#     'cost': cost,
#     'par2': par2,
#     'par3': par3
# })

# # Save the DataFrame as a CSV file
# csv_file = 'monthly_cost_index_data.csv'
# df.to_csv(csv_file, index=False)

# # Load the DataFrame from the CSV file
# df_loaded = pd.read_csv(csv_file, parse_dates=['date'])

# # Plot the 'cost' column with dates
# plt.figure(figsize=(12, 6))
# plt.plot(df_loaded['date'], df_loaded['cost'], marker='o', linestyle='-', color='b', label='Cost')
# plt.xlabel('Date')
# plt.ylabel('Cost')
# plt.title('Monthly Cost Index (2019-2023)')
# plt.legend()
# plt.grid(True)
# plt.show()
