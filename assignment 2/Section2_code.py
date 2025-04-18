# #Perform k-nearest-neighbour classification with all odd values of k between 1 and 13
# (inclusive) using any 4 features in the “*_features.csv” file, briefly justifying your choice of
# features. It is recommended that you use knn package for this section. Report the accuracy
# over the full set of 76 items for each value of k (use all 76 items in this subsection as training
# data and do not worry in this subsection about overfitting to the training data; i.e., do not use
# cross-validation or a separate test set).

#use connected_areas, rows_with_1, no_neight_horiz, custom
#why use them? they have high correlation with our dummy label
# 1 3 5 7 9 11 13

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('40443486_features.csv')

df['dummy_label'] = np.where(df['image_label'].isin(['a','j','sad','smiley','xclaim']), 1,0)

features = df.select_dtypes(include=[np.number]).columns.tolist()  # Select all numeric columns

corr_matrix = df[features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('40443486_features.csv')

df['dummy_label'] = np.where(df['image_label'].isin(['a','j','sad','smiley','xclaim']), 1,0)

X = df[['connected_areas', 'rows_with_1', 'no_neigh_horiz', 'custom']]
y = df['dummy_label']

accuracies = []

for k in [1,3,5,7,9,11,13]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    pred = knn.predict(X)

    # Evaluate accuracy
    acc = accuracy_score(y, pred)
    print(f"k={k} Accuracy: {acc}")
    accuracies.append(acc)

plt.plot([1,3,5,7,9,11,13], accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. k')
plt.show()
########################################################################################################################
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
########################################################################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
#k = 1 is the best

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

for k in [1]:
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
        print(f"Fold accuracy: {accuracy:.4f}")#is cross validation accuracy

    average_accuracy = np.mean(fold_accuracies)
    print(f"\nAverage cross-validated accuracy: {average_accuracy:.4f}")

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])

print(f'Confusion Matrix (All Features):\n', confusion_matrix)

cm_display.plot()
plt.show()

#next--
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
########################################################################################################################
#same thing just use 3.1 and 3.2 then 1-acc for the error and append into 2 different array then plot
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('40443486_features.csv')

df['dummy_label'] = np.where(df['image_label'].isin(['a', 'j', 'sad', 'smiley', 'xclaim']), 1, 0)

X = df[['connected_areas', 'rows_with_1', 'no_neigh_horiz', 'custom']]
features = ['connected_areas', 'rows_with_1', 'no_neigh_horiz', 'custom']
y = df['dummy_label']

accuracies = []
value_no_cross = []
k_values = [1, 3, 5, 7, 9, 11, 13]
inverse_k = [1 / k for k in k_values]

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
value_cross = []

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

        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        # print(f"Fold accuracy: {accuracy:.4f}")#is cross validation accuracy

    average_accuracy = np.mean(fold_accuracies)
    error_rate_cross = 1 - average_accuracy
    value_cross.append(error_rate_cross)
    # print(f"\nAverage cross-validated accuracy: {average_accuracy:.4f}")

print(value_cross)

# test K= 13 to K= 1
plt.figure(figsize=(9, 6))
plt.plot(inverse_k, value_no_cross, color='teal', marker='o', linestyle='solid', label='Training Errors')
plt.plot(inverse_k, value_cross, color='darkorange', marker='o', linestyle='solid', label='Test Errors')

# bayes error rate line
plt.axhline(y=0.135, color='black', linestyle='--', linewidth=2)

plt.xlabel('1/k')
plt.ylabel('Error Rate')
plt.title('Error Rate vs 1/k')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
