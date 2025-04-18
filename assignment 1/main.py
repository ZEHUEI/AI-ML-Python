import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns

df = pd.read_csv('40443486_features.csv')


df['dummy_label'] = np.where(df['image_label'].isin(['a','b','c','d','e','f','g','h','i''j']), 1, 0)
iris_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=False)
train_data = iris_shuffled[:112]
test_data = iris_shuffled[112:]

plt.figure(figsize=(8, 6))
sns.histplot(data=train_data, x='connected_areas', hue='areas', bins=20, alpha=0.5)
plt.show()

X_train = train_data[['connected_areas']]
y_train = train_data['areas']
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.coef_, model.intercept_)

train_preds = (model.predict_proba(X_train)[:, 1] > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, train_preds)
print("Training Accuracy:", train_accuracy)
X_test = test_data[['connected_areas']]
y_test = test_data['areas']
test_preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_preds)
print("Testing Accuracy:", test_accuracy)



x_vals = np.linspace(df['connected_areas'].min(), df['connected_areas'].max(), 1000).reshape(-1, 1)
x_df = pd.DataFrame(x_vals, columns=['connected_areas'])
fitted_curve = model.predict_proba(x_df)[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(train_data['connected_areas'], train_data['areas'],c= train_data['areas'],cmap='viridis', label='Data')
plt.plot(x_vals, fitted_curve, color='orange', label='Fitted Curve')
plt.xlabel('connected_areas')
plt.ylabel('areas')
plt.title('logistic regression Curve')
plt.legend()
plt.show()