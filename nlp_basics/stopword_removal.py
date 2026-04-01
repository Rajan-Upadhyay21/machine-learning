from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("stopwords")

text = "This is a simple example showing how stopword removal works in NLP."

tokens = word_tokenize(text)
stop_words = set(stopwords.words("english"))

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("Original Tokens:")
print(tokens)

print("\nTokens After Stopword Removal:")
print(filtered_tokens)
