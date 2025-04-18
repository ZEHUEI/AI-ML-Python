import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('40443486_features.csv')

label_map = {
    'a': 'class1',
    'j': 'class1',
    'sad': 'class2',
    'smiley': 'class3',
    'xclaim': 'class4'
}
df['multiclass_label'] = df['image_label'].map(label_map)
df = df.dropna(subset=['multiclass_label'])

features = ['connected_areas', 'rows_with_1', 'no_neigh_horiz', 'custom']
X = df[features]
y = df['multiclass_label']

kfolds = 5
kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
k_values = [1, 3, 5, 7, 9, 11, 13]

for k in k_values:
    fold_accuracies = []
    y_test_final, y_pred_final = None, None

    for train_index, test_index in kf.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        X_train = train_data[features]
        y_train = train_data['multiclass_label']
        X_test = test_data[features]
        y_test = test_data['multiclass_label']

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)

        y_test_final = y_test
        y_pred_final = y_pred

    average_accuracy = np.mean(fold_accuracies)
    print(f"k = {k}, Average Accuracy = {average_accuracy:.4f}")

    cm = confusion_matrix(y_test_final, y_pred_final, labels=['class1', 'class2', 'class3', 'class4'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Letters', 'sad', 'smiley', 'xclaim'])

    fig, ax = plt.subplots(figsize=(6, 5))
    cm_display.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f"Confusion Matrix (k={k})")
    plt.tight_layout()
    plt.show()
