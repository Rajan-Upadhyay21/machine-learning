from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

model = LogisticRegression(max_iter=500)
selector = RFE(estimator=model, n_features_to_select=2)

X_new = selector.fit_transform(X, y)
selected_columns = [feature_names[i] for i in selector.get_support(indices=True)]

print("Selected Features:")
print(selected_columns)

print("\nFeature Rankings:")
print(selector.ranking_)

print("\nReduced Dataset:")
print(pd.DataFrame(X_new, columns=selected_columns).head())
