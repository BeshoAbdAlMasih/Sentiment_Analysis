from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Adjusted to reflect the actual labels in your training set (0 and 4)
LABELS = {0: "Negative", 4: "Positive"}

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    dominant_sentiment: str
    confidence: float
    distribution: dict

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    vec = vectorizer.transform([req.text])
    pred = int(model.predict(vec)[0])
    
    # Binary classifiers return a single float from decision_function
    score = model.decision_function(vec)[0]
    
    # Convert the decision score into a probability using the sigmoid function
    prob_positive = 1 / (1 + np.exp(-score))
    prob_negative = 1 - prob_positive

    distribution = {
        "negative": round(float(prob_negative) * 100),
        "positive": round(float(prob_positive) * 100),
    }

    # The confidence is the probability of the predicted class
    confidence = prob_positive if pred == 4 else prob_negative

    return AnalyzeResponse(
        dominant_sentiment=LABELS[pred],
        confidence=round(float(confidence), 3),
        distribution=distribution,
    )

@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")