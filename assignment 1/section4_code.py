import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('40443486_features.csv')

features = ['connected_areas']
X = df[features]
y = df['aspect_ratio']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

#connected_areas, no_neigh_horiz, no_neigh_below, no_neigh_above, ,,custom , rows_with_1
features = ['no_neigh_horiz','no_neigh_below','rows_with_1','nr_pix','no_neigh_right']
X = df[features]
y = df['aspect_ratio']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

###################################################################################################
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns

df = pd.read_csv('40443486_features.csv')

df['dummy_label'] = np.where(df['image_label'].isin(['a','b','c','d','e','f','g','h','i','j']), 0, 1)

iris_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=False)
train_data = iris_shuffled[:112]
test_data = iris_shuffled[112:]

X_train = train_data[['connected_areas']].values
y_train = train_data['dummy_label'].values
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.coef_, model.intercept_)

train_preds = (model.predict_proba(X_train)[:, 1] > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, train_preds)
print("Training Accuracy:", train_accuracy)

X_test = test_data[['connected_areas']].values
y_test = test_data['dummy_label'].values
test_preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_preds)
print("Testing Accuracy:", test_accuracy)


x_vals = np.linspace(df['connected_areas'].min(), df['connected_areas'].max(), 1000).reshape(-1, 1)
fitted_curve = model.predict_proba(x_vals.reshape(-1, 1))[:, 1]

# plt.figure(figsize=(8, 6))
plt.scatter(train_data['connected_areas'], train_data['dummy_label'],c= train_data['dummy_label'],cmap='viridis', label='Data')
plt.plot(x_vals, fitted_curve, color='red', label='Fitted Curve')
plt.xlabel('connected_areas')
plt.ylabel('Probability of connected areas')
plt.title('LOGISTIC REGRESSION CURVE')
plt.legend()
plt.show()
#################################################################
import pandas as pd

df = pd.read_csv('features_test.csv')

features = ['nr_pix', 'aspect_ratio', 'neigh_1']

# Create new categorical features based on median splits
for feature in features:
    median_value = df[feature].median()
    df[f'split{features.index(feature) + 1}'] = (df[feature] > median_value).astype(int)

# Define the three classes
classes = {
    'letter': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
    'face': ['smiley', 'sad'],
    'exclamation mark': ['xclaim']
}

results = []

for class_name, class_labels in classes.items():
    class_data = df[df['image_label'].isin(class_labels)]
    proportions = {
        'Class': class_name,
        'split1': class_data['split1'].mean(),
        'split2': class_data['split2'].mean(),
        'split3': class_data['split3'].mean()
    }
    results.append(proportions)

results_df = pd.DataFrame(results)

print(results_df)