from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="models/bert_model",
    tokenizer="models/bert_model"
)

while True:
    text = input("Enter message (or 'exit'): ")

    if text.lower() == "exit":
        break

    result = classifier(text)[0]
    ml_label = result["label"]
    ml_score = result["score"]

    if ml_label.upper() == "SPAM":
        final = "Spam 🚨 (BERT)"
    else:
        final = "Ham ✅"

    print("\nPrediction:", final)
    print("ML Confidence:", round(ml_score, 4))
    print("-" * 40)
    print(result)