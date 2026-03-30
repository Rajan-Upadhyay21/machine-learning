import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Sample numerical data
data = {
    "x": [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Generate polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(df)

poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df.columns))

print("\nPolynomial Features Dataset:")
print(poly_df)
