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
