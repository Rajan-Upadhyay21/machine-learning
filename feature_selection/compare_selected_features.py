from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

df_original = pd.DataFrame(X, columns=feature_names)
print("Original Dataset:")
print(df_original.head())

selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

selected_columns = [feature_names[i] for i in selector.get_support(indices=True)]
df_selected = pd.DataFrame(X_new, columns=selected_columns)

print("\nSelected Features:")
print(selected_columns)

print("\nReduced Dataset:")
print(df_selected.head())
