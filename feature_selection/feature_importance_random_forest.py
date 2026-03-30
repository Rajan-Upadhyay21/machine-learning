from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

print("Feature Importances from Random Forest:")
print(importance_df.sort_values(by="Importance", ascending=False))
