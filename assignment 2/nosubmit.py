# Nt is [25,75,125,175,225,275,325,375] // Np = {2,4,6,8}
# 5 fold cross validation
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
# Read in features (assuming it’s a feature matrix)

csv_file = 'all_features.csv'
df = pd.read_csv(csv_file, sep='\t', header=None)

nt_values= [25,75,125,175,225,275,325,375]
np_values= [2,4,6,8]

results=[]

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

total =0
kfolds =5
kf = KFold(n_splits=kfolds,shuffle=True,random_state=42)

i=0
j=0
count = 0
amount = 0
total =0
bruh =0

# Perform grid search

for nt in nt_values:
    print(nt)
    for j, np_ in enumerate(np_values):
        print(np_)

        fold_accuracies = []

        rf = RandomForestClassifier(n_estimators=nt, max_features=np_, random_state=42)

        for train_index, test_index in kf.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)

        mean_acc = np.mean(fold_accuracies)
        results.append({'nt': nt, 'np': np_, 'accuracy': mean_acc})
        print(f"Mean accuracy for nt={nt}, np={np_}: {mean_acc:.4f}")
    print('\n')

# # Convert results to DataFrame
# results_df = pd.DataFrame(results)
#
# # Pivot the results for heatmap
# pivot_results = results_df.pivot(index='nt', columns='np', values='accuracy')
#
# # Plot heatmap
# plt.figure(figsize=(10, 6))
# plt.imshow(pivot_results, cmap='viridis', aspect='auto')
# plt.colorbar(label='Accuracy')
# plt.xticks(np.arange(len(np_values)), np_values)
# plt.yticks(np.arange(len(nt_values)), nt_values)
# plt.xlabel('Number of predictors at each node (Np)')
# plt.ylabel('Number of trees (Nt)')
# plt.title('Random Forest Grid Search Results')
# plt.show()