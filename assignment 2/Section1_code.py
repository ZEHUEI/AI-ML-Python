import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


df = pd.read_csv('40443486_features.csv')

df['dummy_label'] = np.where(df['image_label'].isin(['a','b','c','d','e','f','g','h','i','j']), 1,0)


X = df[['nr_pix', 'aspect_ratio']]
y = df['dummy_label']

#shuffle and take 20%
#test =20% train =20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
print("Coefficients:",model.coef_, "Intercept:",model.intercept_)

train_preds = (model.predict_proba(X_train)[:, 1] > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, train_preds)
print("Training Accuracy:", train_accuracy)

test_preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_preds)
print("Testing Accuracy:", test_accuracy)

actual = y_test.to_numpy(dtype=int)#true label from the test
predicted = test_preds.astype(int)#predicted value from the model

confusion_matrix = metrics.confusion_matrix(actual, predicted)
print(confusion_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

Accuracy = metrics.accuracy_score(actual, predicted)
print("Accuracy:" , Accuracy)

Precision = metrics.precision_score(actual, predicted,pos_label=1)
print("Precision:", Precision)

Recall = metrics.recall_score(actual, predicted,pos_label=1)
print("Recall: " , Recall)

F1_score = metrics.f1_score(actual, predicted)
print("F1-score:", F1_score)

#check F1 score with formula
print(2*(Precision*Recall)/(Precision+Recall))


cm_display.plot()
plt.show()
########################################################################################################################
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
########################################################################################################################
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

df = pd.read_csv('40443486_features.csv')

df['dummy_label'] = np.where(df['image_label'].isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']), 1, 0)

features = ['nr_pix' , 'aspect_ratio']
X = df[['nr_pix', 'aspect_ratio']]
y = df['dummy_label']

model = LogisticRegression()
model.fit(X, y)

df['predicted_val'] = model.predict_proba(X)[:, 1]

df['predicted_class'] = np.where(df['predicted_val'] > 0.5, 1, 0)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy: {cv_scores.mean()}')

fpr, tpr, thresholds = roc_curve(y, df['predicted_val'])
roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc}')

for threshold in [0.4, 0.5, 0.6]:
    df['predicted_class'] = np.where(df['predicted_val'] >= threshold, 1, 0)
    cm = confusion_matrix(y, df['predicted_class'])
    print(f'\nConfusion Matrix (Threshold = {threshold}):\n', cm)
    print('Sensitivity and Specificity:', classification_report(y, df['predicted_class'], target_names=['nr_pix', 'aspect_ratio']))

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
