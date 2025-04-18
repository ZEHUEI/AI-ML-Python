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


