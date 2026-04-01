positive_words = ["good", "great", "excellent", "amazing", "happy", "love"]
negative_words = ["bad", "terrible", "awful", "sad", "hate", "worst"]

text = "I love this product because it is amazing and excellent"

tokens = text.lower().split()

positive_count = sum(1 for word in tokens if word in positive_words)
negative_count = sum(1 for word in tokens if word in negative_words)

print("Positive Word Count:", positive_count)
print("Negative Word Count:", negative_count)

if positive_count > negative_count:
    print("Sentiment: Positive")
elif negative_count > positive_count:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")
