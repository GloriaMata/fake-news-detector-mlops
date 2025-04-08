import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import joblib

# 1. Cargar el dataset
df = pd.read_csv("fake_news_dataset.csv")  # Ajusta el path si lo corres localmente

# 2. Definir target y features
X = df.drop(columns=["label", "id", "title", "date_published", "text"])
y = df["label"]
text_data = df["text"]

# 3. Separación de datos
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X, y, text_data, test_size=0.2, random_state=42, stratify=y
)

# 4. Columnas categóricas y numéricas
cat_cols = ["author", "state", "source", "category", "political_bias", "fact_check_rating"]
num_cols = [col for col in X.columns if col not in cat_cols]

# 5. Preprocesamiento por tipo de dato
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),
    ("text", TfidfVectorizer(max_features=1000, stop_words="english"), "text")
])

# 6. Combinar texto con X
X_train_full = X_train.copy()
X_train_full["text"] = text_train
X_test_full = X_test.copy()
X_test_full["text"] = text_test

# 7. Pipeline completo
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

# 8. Búsqueda de hiperparámetros
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", verbose=1)
grid_search.fit(X_train_full, y_train)

# 9. Evaluación
y_pred = grid_search.predict(X_test_full)
print(classification_report(y_test, y_pred))

# 10. Guardar el modelo
joblib.dump(grid_search.best_estimator_, "fake_news_classifier.joblib")
