from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

documents = [
    "I love machine learning",
    "Machine learning is powerful",
    "I love Python programming"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print(df)
