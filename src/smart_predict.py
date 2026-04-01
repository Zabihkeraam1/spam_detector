import joblib
import re

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

spam_keywords = ["free", "win", "cash", "prize", "urgent", "offer"]

def count_links(text):
    return len(re.findall(r"http[s]?://", text))

def count_exclamations(text):
    return text.count("!")

def has_repeated_chars(text):
    return bool(re.search(r"(.)\1{3,}", text))

def keyword_score(text):
    text = text.lower()
    return sum(word in text for word in spam_keywords)

def smart_predict(message):
    tfidf = vectorizer.transform([message])
    ml_pred = model.predict(tfidf)[0]

    score = 0

    if count_links(message) > 0:
        score += 2
    if count_exclamations(message) > 3:
        score += 1
    if has_repeated_chars(message):
        score += 1
    if keyword_score(message) >= 2:
        score += 2

    if ml_pred == 1 or score >= 3:
        return "Spam 🚨"
    else:
        return "Ham ✅"

if __name__ == "__main__":
    print("🤖 Smart Spam Detector (type 'exit' to quit)\n")

    while True:
        msg = input("Enter message: ")

        if msg.lower() == "exit":
            break

        result = smart_predict(msg)
        print("Prediction:", result)
        print("-" * 40)