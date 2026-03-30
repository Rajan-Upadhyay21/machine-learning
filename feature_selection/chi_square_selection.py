from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(X_scaled, y)

selected_columns = [feature_names[i] for i in selector.get_support(indices=True)]

print("Selected Features:")
print(selected_columns)

print("\nChi-Square Scores:")
print(selector.scores_)

print("\nReduced Dataset:")
print(pd.DataFrame(X_new, columns=selected_columns).head())
