import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Data/Processed/Engine_Oil.csv")

# Separate features and target
X = df.drop("engine_oil", axis=1)
y = df["engine_oil"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Define Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

results = []

# -----------------------------
# Model Training & Evaluation
# -----------------------------
for name, model in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    # 5-Fold Cross Validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    cv_mean = np.mean(cv_scores)

    results.append([name, acc, f1, roc, cv_mean])

# -----------------------------
# Create Results DataFrame
# -----------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "F1 Score", "ROC-AUC", "CV Accuracy"]
)

print("\nModel Comparison Results:\n")
print(results_df)

# -----------------------------
# Save Results
# -----------------------------
os.makedirs("Results", exist_ok=True)
results_df.to_csv("Results/model_comparison_results.csv", index=False)

print("\nResults saved to Results/model_comparison_results.csv")