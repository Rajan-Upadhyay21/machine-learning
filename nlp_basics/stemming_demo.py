from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

text = "playing played plays runner running easily"
tokens = word_tokenize(text)

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

print("Original Words:")
print(tokens)

print("\nStemmed Words:")
print(stemmed_words)
