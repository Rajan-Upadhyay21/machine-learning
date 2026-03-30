from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

scores = mutual_info_classif(X, y, random_state=42)

mi_df = pd.DataFrame({
    "Feature": feature_names,
    "Mutual Information Score": scores
})

print("Mutual Information Scores:")
print(mi_df.sort_values(by="Mutual Information Score", ascending=False))
