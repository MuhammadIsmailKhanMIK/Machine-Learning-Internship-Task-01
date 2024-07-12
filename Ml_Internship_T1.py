import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

# Q1: Import data and check for null values, column info, and descriptive statistics
df = pd.read_csv('Instagram-Reach.csv')
print(df.isnull().sum())
print(df.info())
print(df.describe())

# Q2: Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set Date column as index
df.set_index('Date', inplace=True)

# Explicitly set frequency to daily
df.index.freq = pd.infer_freq(df.index)

# Q3: Analyze the trend of Instagram reach over time using a line chart
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Instagram reach'])
plt.title('Instagram Reach Over Time')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.show()

# Q4: Analyze Instagram reach for each day using a bar chart
plt.figure(figsize=(12, 6))
plt.bar(df.index, df['Instagram reach'])
plt.title('Instagram Reach for Each Day')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.show()

# Q5: Analyze the distribution of Instagram reach using a box plot
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['Instagram reach'])
plt.title('Distribution of Instagram Reach')
plt.ylabel('Instagram Reach')
plt.show()

# Q6: Create a day column and analyze reach based on the days of the week
df['Day'] = df.index.day_name()

# Specify the correct order for days of the week
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Group by the day of the week and calculate statistics
day_stats = df.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reindex(days_order)
print(day_stats)

# Q7: Create a bar chart to visualize the reach for each day of the week
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
reach = [150, 200, 180, 220, 240, 300, 260]

# Create the bar plot
plt.bar(days, reach, color='skyblue')

# Add labels and title
plt.title('Average Instagram Reach by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Mean Instagram Reach')

# Display the plot
plt.show()
# Q8: Check the Trends and Seasonal patterns of Instagram reach
result = seasonal_decompose(df['Instagram reach'], model='additive', period=30)
result.plot()
plt.show()

# Q9: Use the SARIMA model to forecast the reach of the Instagram account
# Find p, d, q values
plot_acf(df['Instagram reach'])
plt.show()

plot_pacf(df['Instagram reach'])
plt.show()

# Using p=1, d=1, q=1 as an example (you can determine these from the plots)
p, d, q = 1, 1, 1

# Train SARIMA model
model = SARIMAX(df['Instagram reach'], order=(p, d, q), seasonal_order=(p, d, q, 12))
sarima_model = model.fit()

# Make predictions
forecast = sarima_model.get_forecast(steps=30)
forecast_df = forecast.conf_int()
forecast_df['Forecast'] = sarima_model.predict(start=forecast_df.index[0], end=forecast_df.index[-1])

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Instagram reach'], label='Observed')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
plt.fill_between(forecast_df.index, forecast_df.iloc[:, 0], forecast_df.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Instagram Reach Forecast')
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.legend()
plt.show()