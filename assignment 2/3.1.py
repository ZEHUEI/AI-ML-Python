# Nt is [25,75,125,175,225,275,325,375] // Np = {2,4,6,8}
# 5 fold cross validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

csv_file = 'all_features.csv'
df = pd.read_csv(csv_file, sep='\t', header=None)

nt_values= [25,75,125,175,225,275,325,375]
np_values= [2,4,6,8]


X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

kfolds =5
kf = KFold(n_splits=kfolds,shuffle=True,random_state=42)

results = []

for nt in nt_values:
    print(nt)
    for np_ in np_values:
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

best_result = max(results, key=lambda x: x['accuracy'])
print('\nBest accuracy and its corresponding nt and np:')
print(f"Best accuracy: {best_result['accuracy']:.4f} for nt={best_result['nt']} and np={best_result['np']}")

results_df = pd.DataFrame(results)

heatmap_data = results_df.pivot(index='np', columns='nt', values='accuracy')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")
plt.title("Random Forest CV Accuracy Heatmap")
plt.xlabel("Number of Trees (Nt)")
plt.ylabel("Number of Predictors (Np)")
plt.tight_layout()
plt.show()