import pandas as pd

from Ml_Internship_Task_1.Ml_Internship_T1 import day_stats
import seaborn as sns

# Read the CSV file
df = pd.read_csv('Instagram-Reach.csv')

# Display the first few rows of the DataFrame
print(df.head())
new_var = sns.barplot(x=day_stats.index, y=day_stats['mean'], yerr=day_stats['std'].values, capsize=0.1)