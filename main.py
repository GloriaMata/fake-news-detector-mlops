from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Cargar modelo entrenado
model = joblib.load("fake_news_classifier.joblib")


# Modelo de entrada
class NewsFeatures(BaseModel):
    author: str
    state: str
    source: str
    category: str
    political_bias: str
    fact_check_rating: str
    sentiment_score: float
    num_shares: int
    source_reputation: float
    is_satirical: int
    plagiarism_score: float
    char_count: int
    has_videos: int
    word_count: int
    has_images: int
    readability_score: float
    clickbait_score: float
    trust_score: float
    num_comments: int
    text: str


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
