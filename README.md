# Sentix API — Twitter Sentiment Analysis

> A production-ready sentiment analysis REST API built with FastAPI, scikit-learn, and a clean dark-themed dashboard UI. Trained on the Sentiment140 dataset — 1.6 million real tweets.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-Sentiment140-blue?style=flat-square&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What It Does

Sentix takes raw tweet text and returns:

- **Dominant sentiment** — Positive or Negative
- **Confidence score** — how certain the model is
- **Distribution** — percentage split across both classes
- **Extracted entities** — key terms detected in the input

---

## Demo

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Just got promoted at work, best day ever!"}'
```

```json
{
  "dominant_sentiment": "Positive",
  "confidence": 0.891,
  "distribution": {
    "positive": 89,
    "negative": 11
  }
}
```

---

## Project Structure

```
Sentiment_analsis/
├── main.py                  # FastAPI application
├── model.pkl                # Trained LinearSVC model
├── vectorizer.pkl           # Fitted TF-IDF vectorizer
├── requirements.txt
├── train.ipynb              # Full training notebook
└── static/
    ├── WebviewDesktop.html  # Dashboard UI (desktop)
```

---

## Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| ML model | LinearSVC (scikit-learn) |
| Vectorizer | TF-IDF · 50k features · bigrams |
| Validation | Pydantic v2 |
| Frontend | HTML + Tailwind CSS + Vanilla JS |
| Dataset | Sentiment140 — 1.6M tweets (Kaggle) |

---

## Dataset

**Sentiment140** by Kazanova — [kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

- 1,599,999 tweets collected via Twitter API in 2009
- Binary labels: `0` = Negative, `4` = Positive (no neutral class)
- Automatically labelled using tweet emoticons as weak supervision
- 80/20 train/test split → 1.28M training rows, 320k test rows

**Preprocessing applied:**
- Lowercasing
- URL and mention removal
- Punctuation stripping
- TF-IDF vectorization with unigrams + bigrams (`ngram_range=(1,2)`)
- 50,000 maximum features

---

## Model Performance

Evaluated on **320,000 held-out tweets**.

### LogisticRegression (`max_iter=1500`)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative (0) | 0.81 | 0.83 | 0.82 | 156,064 |
| Positive (4) | 0.83 | 0.81 | 0.82 | 163,936 |
| **Accuracy** | | | **0.82** | 320,000 |

### LinearSVC ✅ — deployed model

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative (0) | 0.80 | 0.82 | 0.81 | 155,526 |
| Positive (4) | 0.83 | 0.81 | 0.82 | 164,474 |
| **Accuracy** | | | **0.82** | 320,000 |

Both models reach **82% accuracy** on 320k unseen tweets. LinearSVC was chosen for deployment due to faster inference at serving time.
---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/BeshoAbdAlMasih/sentix-api.git
cd sentix-api
```

### 2. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

### 3. Download the dataset and train (optional — pkl files included)

```python
import kagglehub
path = kagglehub.dataset_download("kazanova/sentiment140")
```

Then open `train.ipynb` and run all cells. This generates `model.pkl` and `vectorizer.pkl`.

### 4. Run the API

```bash
uvicorn main:app --reload
```

API runs at `http://127.0.0.1:8000`

### 5. Open the dashboard

```
http://127.0.0.1:8000/static/WebviewDesktop.html
```

---

## API Reference

### `POST /analyze`

Analyze the sentiment of a text input.

**Request body**
```json
{
  "text": "Your tweet text here"
}
```

**Response**
```json
{
  "dominant_sentiment": "Positive",
  "confidence": 0.891,
  "distribution": {
    "positive": 89,
    "negative": 11
  }
}
```

**Error responses**

| Code | Reason |
|---|---|
| 422 | Missing or invalid request body |
| 500 | Internal model error |

---

### `GET /health`

```json
{ "status": "ok" }
```



## License

MIT
