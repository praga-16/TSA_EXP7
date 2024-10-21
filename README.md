# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python in Gold Price Prediction.
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Step 2: Read the CSV file into a DataFrame
data = pd.read_csv('Gold Price Prediction.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Assuming 'Price Today' is the column of interest for gold price
gold_prices = data[['Date', 'Price Today']]

# Set 'Date' as the index
gold_prices.set_index('Date', inplace=True)

# Step 3: Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(gold_prices['Price Today'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Step 4: Split the data into training and testing sets (80-20 split)
train_size = int(len(gold_prices) * 0.8)
train_data, test_data = gold_prices[0:train_size], gold_prices[train_size:]

# Step 5: Fit an AutoRegressive (AR) model with 13 lags
model = AutoReg(train_data['Price Today'], lags=13)
ar_model_fit = model.fit()

# Step 6: Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plt.figure(figsize=(10, 5))
plot_acf(train_data['Price Today'], lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(train_data['Price Today'], lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Step 7: Make predictions using the AR model
predictions = ar_model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# Step 8: Compare the predictions with the test data
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Actual Test Data', color='blue')
plt.plot(test_data.index, predictions, label='Predictions', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Gold Prices')
plt.xlabel('Date')
plt.ylabel('Price Today')
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)
```
### OUTPUT:

## GIVEN DATA:
## AUGMENTED DICKEY-FULLER TEST:

![image](https://github.com/user-attachments/assets/a4254eba-38c3-41eb-86ac-d85080b959ff)

## PACF - ACF
![image](https://github.com/user-attachments/assets/9f58d032-0220-46a6-abb0-dcdfe460f731)
![image](https://github.com/user-attachments/assets/c1536afe-c8f8-419f-abdf-82c216b53641)


## PREDICTION
## FINIAL PREDICTION
![image](https://github.com/user-attachments/assets/87d153a5-cd3d-403c-b460-67b58bdcbc49)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
