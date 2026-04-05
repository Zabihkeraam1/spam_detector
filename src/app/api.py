from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = FastAPI()

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam API working 🚀"}

@app.post("/predict")
def predict(msg: Message):
    text = msg.text

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vector)[0]
        confidence = float(max(prob))
    else:
        confidence = None

    return {
        "text": text,
        "prediction": "Spam 🚨" if prediction == 1 else "Ham ✅",
        "confidence": confidence
    }