from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI()

# Cargar modelo entrenado
model = joblib.load("fake_news_classifier.joblib")


# Modelo de entrada
class NewsFeatures(BaseModel):
    author: str = Field(..., min_length=1)
    state: str
    source: str
    category: str
    political_bias: str
    fact_check_rating: str
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    num_shares: int = Field(..., ge=0)
    source_reputation: float = Field(..., ge=0.0, le=1.0)
    is_satirical: int = Field(..., ge=0, le=1)
    plagiarism_score: float = Field(..., ge=0.0, le=1.0)
    char_count: int = Field(..., ge=0)
    has_videos: int = Field(..., ge=0, le=1)
    word_count: int = Field(..., ge=0)
    has_images: int = Field(..., ge=0, le=1)
    readability_score: float
    clickbait_score: float = Field(..., ge=0.0, le=1.0)
    trust_score: float = Field(..., ge=0.0, le=1.0)
    num_comments: int = Field(..., ge=0)
    text: str = Field(..., min_length=10)


@app.get("/")
def home():
    return {"message": "Fake News Classifier API is running."}


@app.post("/predict")
def predict_news(news: NewsFeatures):
    # Convertir entrada en DataFrame
    input_df = pd.DataFrame([news.dict()])

    # Predecir
    prediction = model.predict(input_df)[0]
    return {"prediction": prediction}
