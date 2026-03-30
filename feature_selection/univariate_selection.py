from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

selected_columns = [feature_names[i] for i in selector.get_support(indices=True)]

print("Selected Features:")
print(selected_columns)

print("\nScores:")
print(selector.scores_)

print("\nReduced Dataset:")
print(pd.DataFrame(X_new, columns=selected_columns).head())
