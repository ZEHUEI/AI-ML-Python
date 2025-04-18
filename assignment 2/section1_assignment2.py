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