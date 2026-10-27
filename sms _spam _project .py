
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the SMS dataset
data = pd.read_csv("train.csv")

X = data["sms"].fillna("").astype(str)
y = data["label"].astype(int)

print("Dataset loaded successfully!")
print("Total SMS messages:", len(data))

# 2. Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Convert SMS text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Create the four machine-learning models
models = {
    "M1 - Multinomial Naive Bayes": MultinomialNB(),
    "M2 - Logistic Regression": LogisticRegression(max_iter=1000),
    "M3 - Linear SVM": LinearSVC(),
    "M4 - Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

results = {}

# 5. Train and evaluate each model
for name, model in models.items():
    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

# 6. Compare the four models
print("\n========== FINAL MODEL COMPARISON ==========")

for name, accuracy in sorted(
    results.items(), key=lambda item: item[1], reverse=True
):
    print(name, ":", round(accuracy * 100, 2), "%")

best_model = max(results, key=results.get)

print("\nHighest accuracy model:", best_model)
print("Highest accuracy:", round(results[best_model] * 100, 2), "%")

# 7. Predict whether a new SMS is spam

best_model_object = models[best_model]

while True:
    message = input("\nEnter an SMS (or type exit to stop): ")

    if message.lower() == "exit":
        break

    message_tfidf = vectorizer.transform([message])
    prediction = best_model_object.predict(message_tfidf)

    if prediction[0] == 1:
        print("Spam SMS")
    else:
        print("Not Spam SMS")

