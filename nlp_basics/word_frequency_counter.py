from collections import Counter

text = "machine learning is powerful and machine learning is useful"

tokens = text.lower().split()
word_counts = Counter(tokens)

print("Word Frequencies:")
print(word_counts)
