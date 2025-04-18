import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate data for nr_pix based on mean, median, and std
def simulate_data(mean, median, std, size=100):
    # Generate random data with the given mean and std
    data = np.random.normal(loc=mean, scale=std, size=size)
    # Adjust the median by shifting the data
    data = data - np.median(data) + median
    return data

# Non-letters data
non_letters_mean = 3.333333
non_letters_median = 4
non_letters_std = 0.950765
non_letters_data = simulate_data(non_letters_mean, non_letters_median, non_letters_std)

# Letters data
letters_mean = 1.2
letters_median = 1
letters_std = 0.402524
letters_data = simulate_data(letters_mean, letters_median, letters_std)

# Combine into a list for boxplot
data = [non_letters_data, letters_data]
labels = ['Non-Letters', 'Letters']

# Create the boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue'), showmeans=True, meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black'))

# # Overlay data points
# for i, d in enumerate(data):
#     x = np.random.normal(i + 1, 0.04, size=len(d))  # Add jitter for better visibility
#     plt.plot(x, d, 'k.', alpha=0.5)

# Customize the plot
plt.title('Boxplot of connected_areas for Non-Letters and Letters')
plt.xlabel('Category')
plt.ylabel('connected_areas')
plt.tight_layout()
plt.show()