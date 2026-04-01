from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

text = "The striped bats are hanging on their feet for best"
tokens = word_tokenize(text)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

print("Original Words:")
print(tokens)

print("\nLemmatized Words:")
print(lemmatized_words)
