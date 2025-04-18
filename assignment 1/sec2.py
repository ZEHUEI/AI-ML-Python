import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv('40443486_features.csv')

# Step 2: Create a histogram using Matplotlib
plt.hist(data['nr_pix'], bins=20, color='pink', edgecolor='black', density=True)

# Step 3: Add a KDE curve
kde = gaussian_kde(data['nr_pix'])
x_values = np.linspace(data['nr_pix'].min(), data['nr_pix'].max(), 1000)
plt.plot(x_values, kde(x_values), color='red', label='KDE')

# Step 4: Add labels and title
plt.title('Distribution of nr_pix')
plt.xlabel('Number of Black Pixels (nr_pix)')
plt.ylabel('Density')
plt.legend()
plt.show()