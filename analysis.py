import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/exercise.csv"
df = pd.read_csv(url)
# --- DATA CLEANING ---

# Remove missing values
df = df.dropna()

# Remove unrealistic pulse values (outliers)
df = df[(df['pulse'] > 40) & (df['pulse'] < 200)]

# Reset index after cleaning
df = df.reset_index(drop=True)

print("Data cleaned successfully")
print(df.head())
print(df.describe())

df['pulse'].plot(kind='hist', title='Pulse Distribution')
plt.show()

print("Average pulse:", df['pulse'].mean())
# --- LEVEL 2 ANALYSIS ---

# Convert time to numbers
df['time_min'] = df['time'].str.replace(' min', '').astype(int)

# Scatter plot (relationship between time and pulse)
df.plot(x='time_min', y='pulse', kind='scatter', title='Pulse vs Time')
plt.show()

# Correlation
correlation = df['time_min'].corr(df['pulse'])
print("Correlation between time and pulse:", correlation)
# --- LEVEL 3 ANALYSIS ---

# Group by time and calculate average pulse
avg_by_time = df.groupby('time_min')['pulse'].mean()
print(avg_by_time)

# Plot average pulse over time
avg_by_time.plot(kind='line', title='Average Pulse by Time')
plt.show()


X = df[['time_min']]
y = df['pulse']

model = LinearRegression()
model.fit(X, y)

predicted_pulse = model.predict([[20]])
print("Predicted pulse at 20 minutes:", predicted_pulse[0])

# Plot regression line
plt.scatter(df['time_min'], df['pulse'], label='Data')

# Prediction line
x_range = np.array(df['time_min']).reshape(-1, 1)
y_pred = model.predict(x_range)

plt.plot(df['time_min'], y_pred, color='red', label='Regression Line')

plt.title('Regression: Time vs Pulse')
plt.xlabel('Time (min)')
plt.ylabel('Pulse')
plt.legend()

plt.show()