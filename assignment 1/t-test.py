import pandas as pd
from scipy import stats

# Load the data
data = pd.read_csv('40443486_features.csv')

# Separate the data into letters and non-letters
letters = data[data['image_label'].isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])]
non_letters = data[data['image_label'].isin(['sad', 'smiley', 'xclaim'])]

# Perform the t-test
t_stat, p_value = stats.ttest_ind(letters['custom'], non_letters['custom'])

# Output the results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Determine significance
alpha = 0.05
if p_value < alpha:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")