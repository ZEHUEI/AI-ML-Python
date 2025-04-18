import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('40443486_features.csv')

df['dummy_label'] = np.where(df['image_label'].isin(['a','j','sad','smiley','xclaim']), 1,0)

X = df[['connected_areas', 'rows_with_1', 'no_neigh_horiz', 'custom']]
features = ['connected_areas', 'rows_with_1', 'no_neigh_horiz', 'custom']
y = df['dummy_label']

accuracies = []
value_no_cross =[]
k_values = [1, 3, 5, 7, 9, 11, 13]
inverse_k = [1/k for k in k_values]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    pred = knn.predict(X)

    acc = accuracy_score(y, pred)

    accuracies.append(acc)
    error_rate_nocross = 1 - acc
    value_no_cross.append(error_rate_nocross)

print(value_no_cross)

accuracies = []
value_cross =[]

kfolds = 5
kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

k_neighbors = 5
accuracy_scores = []

for k in k_values:
    fold_accuracies = []
    # print("\n")
    # print("k=",k)

    for train_index, test_index in kf.split(df):  # train = test = 20%
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        X_train = train_data[features]
        y_train = train_data['dummy_label']
        X_test = test_data[features]
        y_test = test_data['dummy_label']

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test , y_pred)
        fold_accuracies.append(accuracy)
        # print(f"Fold accuracy: {accuracy:.4f}")#is cross validation accuracy

    average_accuracy = np.mean(fold_accuracies)
    error_rate_cross = 1 - average_accuracy
    value_cross.append(error_rate_cross)
    # print(f"\nAverage cross-validated accuracy: {average_accuracy:.4f}")

print(value_cross)

#test K= 13 to K= 1
plt.figure(figsize=(9, 6))
plt.plot(inverse_k, value_no_cross, color='teal', marker='o', linestyle='solid', label='Training Errors')
plt.plot(inverse_k, value_cross, color='darkorange', marker='o', linestyle='solid', label='Test Errors')

#bayes error rate line
plt.axhline(y=0.135, color='black', linestyle='--', linewidth=2)

plt.xlabel('1/k')
plt.ylabel('Error Rate')
plt.title('Error Rate vs 1/k')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
