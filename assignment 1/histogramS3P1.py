import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = "40443486_features.csv"
df = pd.read_csv(file_path)

# Display basic info and first few rows
df.info(), df.head()


# Drop non-numeric columns
numeric_df = df.drop(columns=["image_label", "index"])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Extract correlation pairs and sort
correlation_unstacked = correlation_matrix.unstack()
sorted_correlations = correlation_unstacked.abs().sort_values(ascending=False)

# Remove self-correlations
sorted_correlations = sorted_correlations[sorted_correlations < 1].drop_duplicates()

# Display top correlated feature pairs
top_correlations = sorted_correlations.head(10)
top_correlations