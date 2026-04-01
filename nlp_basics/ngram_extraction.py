from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

documents = [
    "machine learning is fun",
    "learning python is useful"
]

vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(documents)

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print(df)
