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

# PREPROCESSING
# --------------------------------------------

y = df["label”]
X = df.drop(["id",  “label”], axis=1)
df = df.dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --------------------------------------------
# PIPELINE, normaliza los datos y utilizamos modelo de regresión logística
# --------------------------------------------
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# --------------------------------------------
# HYPERPARAMETER OPTIMIZATION
# --------------------------------------------
param_grid_logreg = {
    'classifier__C': [0.1, 1, 10, 100]
}

# --------------------------------------------
# TRAINING
# --------------------------------------------
grid_search_log_reg = GridSearchCV(pipeline, param_grid_logreg, cv=5, n_jobs=-1)
grid_search_log_reg.fit(X_train, y_train)

# --------------------------------------------
# BEST MODEL SELECTION
# --------------------------------------------
best_params = grid_search_log_reg.best_params_
best_model = grid_search_log_reg.best_estimator_
coefficients = best_model.named_steps['classifier'].coef_[0]
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients, color='skyblue')
plt.xlabel('Coeficiente')
plt.title('Importancia de las características en el modelo de regresión logística')
plt.savefig("feature_importance.png", dpi=120)
plt.close()

# --------------------------------------------
# METRICS
# --------------------------------------------
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

with open("metrics.txt", 'w') as outfile:
    outfile.write("Training accuracy: %2.1f%%\n" % accuracy)

# --------------------------------------------
# SERIALIZING
# --------------------------------------------
model_filename = 'model/logistic_regression_model.pkl'
joblib.dump(best_model, model_filename)

print("----- The train process finished -----")

