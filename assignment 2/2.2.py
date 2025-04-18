import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('40443486_features.csv')

df['dummy_label'] = np.where(df['image_label'].isin(['a','j','sad','smiley','xclaim']), 1,0)

X = df[['connected_areas', 'rows_with_1', 'no_neigh_horiz', 'custom']]
features = ['connected_areas', 'rows_with_1', 'no_neigh_horiz', 'custom']
y = df['dummy_label']

accuracies = []

kfolds = 5
kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

k_neighbors = 5
accuracy_scores = []

for k in [1,3,5,7,9,11,13]:
    fold_accuracies = []
    print("\n")
    print("k=",k)

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
        print(f"Fold accuracy: {accuracy:.4f}")

    average_accuracy = np.mean(fold_accuracies)
    print(f"\nAverage cross-validated accuracy: {average_accuracy:.4f}")