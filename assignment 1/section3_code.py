import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np

data = pd.read_csv('40443486_features.csv')

plt.hist(data['nr_pix'], bins=20, color='pink', edgecolor='black', density=True)

kde = gaussian_kde(data['nr_pix'])
x_values = np.linspace(data['nr_pix'].min(), data['nr_pix'].max(), 1000)
plt.plot(x_values, kde(x_values), color='red', label='KDE')

plt.title('Distribution of nr_pix')
plt.xlabel('Number of Black Pixels (nr_pix)')
plt.ylabel('Density')
plt.legend()
plt.show()

##########################################################################
import pandas as pd

data = pd.read_csv('40443486_features.csv')

letters = data[data['image_label'].isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])]
non_letters = data[~data['image_label'].isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])]

letters_summary = letters.describe().loc[['mean', '50%', 'std']]
non_letters_summary = non_letters.describe().loc[['mean', '50%', 'std']]

letters_summary.rename(index={'50%': 'median'}, inplace=True)
non_letters_summary.rename(index={'50%': 'median'}, inplace=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print("Summary Statistics for Letters:")
print(letters_summary)

print("\nSummary Statistics for Non-Letters:")
print(non_letters_summary)

##########################################################################
import pandas as pd
from scipy import stats

data = pd.read_csv('40443486_features.csv')

letters = data[data['image_label'].isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])]
non_letters = data[data['image_label'].isin(['sad', 'smiley', 'xclaim'])]

#t-tests
t_stat, p_value = stats.ttest_ind(letters['custom'], non_letters['custom'])

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")
##################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file_path = "40443486_features.csv"
df = pd.read_csv(file_path)

df.info(), df.head()

numeric_df = df.drop(columns=["image_label", "index"])

correlation_matrix = numeric_df.corr()

#heat map
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

correlation_unstacked = correlation_matrix.unstack()
sorted_correlations = correlation_unstacked.abs().sort_values(ascending=False)

sorted_correlations = sorted_correlations[sorted_correlations < 1].drop_duplicates()

top_correlations = sorted_correlations.head(10)
top_correlations
