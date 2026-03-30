from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

emails = [
    "Win a free iPhone now",
    "Claim your lottery prize today",
    "Meeting scheduled for tomorrow",
    "Please review the project report",
    "Congratulations you won a cash reward",
    "Let us discuss the assignment",
    "Free vacation offer just for you",
    "Team lunch at 1 PM",
    "Earn money quickly from home",
    "Project deadline is next week"
]

labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Spam Email Detection Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
