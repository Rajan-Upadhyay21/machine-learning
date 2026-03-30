from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Convert to DataFrame
df_lda = pd.DataFrame(X_lda, columns=["LD1", "LD2"])

print("Original Shape:", X.shape)
print("Reduced Shape:", X_lda.shape)
print("\nLDA Output:")
print(df_lda.head())
