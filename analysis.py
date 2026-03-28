import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/exercise.csv"
df = pd.read_csv(url)

print(df.head())
print(df.describe())

df['pulse'].plot(kind='hist', title='Pulse Distribution')
plt.show()

print("Average pulse:", df['pulse'].mean())