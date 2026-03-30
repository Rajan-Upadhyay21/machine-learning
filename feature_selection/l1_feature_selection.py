from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=500)
selector = SelectFromModel(model)
selector.fit(X, y)

selected_columns = [feature_names[i] for i in selector.get_support(indices=True)]
X_new = selector.transform(X)

print("Selected Features using L1 Regularization:")
print(selected_columns)

print("\nReduced Dataset:")
print(pd.DataFrame(X_new, columns=selected_columns).head())
