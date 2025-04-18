# Repeat 1.1, but this time use 5-fold crossvalidation (do crossvalidation over all 140 items;
# there is no need to have a separate test set as in Section 1.1). Report the crossvalidated accuracy,
# true positive rate, false positive rate, precision, recall and F1-score.


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('40443486_features.csv')

df['dummy_label'] = np.where(df['image_label'].isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']), 1, 0)

features = ['nr_pix' , 'aspect_ratio']

# shuffle and take 20%
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kfolds = 5
kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

k_neighbors = 5
accuracy_scores = []

for train_index, test_index in kf.split(df):#train = test = 20%
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

    X_train = train_data[features]
    y_train = train_data['dummy_label']
    X_test = test_data[features]
    y_test = test_data['dummy_label']

    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = knn.score(X_test, y_test)
    accuracy_scores.append(accuracy)
    print(f"Fold accuracy: {accuracy:.4f}")

average_accuracy = np.mean(accuracy_scores)
print(f"\nAverage cross-validated accuracy: {average_accuracy:.4f}")

scaler = StandardScaler()
df[['nr_pix', 'aspect_ratio']] = scaler.fit_transform(df[['nr_pix', 'aspect_ratio']])

accuracy_scores_scaled = []

for train_index, test_index in kf.split(df):
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

    X_train = train_data[features]
    y_train = train_data['dummy_label']
    X_test = test_data[features]
    y_test = test_data['dummy_label']

    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)
    accuracy_scores_scaled.append(accuracy)
    print(f"Fold accuracy with scaled features: {accuracy:.4f}")

average_accuracy_scaled = np.mean(accuracy_scores_scaled)
print(f"\nAverage cross-validated accuracy with scaled features: {average_accuracy_scaled:.4f}")

param_grid = {'n_neighbors': list(range(1, 60, 2))}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kfolds)
grid_search.fit(df[features], df['dummy_label'])

print("\nBest parameters (using grid search):", grid_search.best_params_)
print("Best cross-validated accuracy:", grid_search.best_score_)


actual = y_test.to_numpy(dtype=int)  # true label from the test
predicted = y_pred.astype(int)  # predicted value from the model

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])

Accuracy = metrics.accuracy_score(actual, predicted)
print("Accuracy:", Accuracy)

Precision = metrics.precision_score(actual, predicted)
print("Precision:", Precision)

Recall = metrics.recall_score(actual, predicted)
print("Recall: ", Recall)

F1_score = metrics.f1_score(actual, predicted)
print("F1-score:", F1_score)

cm_display.plot()
plt.show()