import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler,
    OneHotEncoder
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_applications.csv"

def evaluate_model(model, X, y, model_name, dataset_name):
    import numpy as np
    # Make predictions
    y_pred = model.predict(X)

    # Get probability of default
    defaulted_index = list(model.classes_).index(1)
    y_prob = model.predict_proba(X)[:, defaulted_index]
    y_pred = (y_prob >= 0.45).astype(int)
    # m = {1:"defaulted", 0:"repaid"}
    # y_pred = np.array([m[x] for x in y_pred])
    # Calculate metrics
    metrics = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(
            y, y_pred,
            pos_label=1,
            zero_division=0
        ),
        "Recall": recall_score(
            y, y_pred,
            pos_label=1,
            zero_division=0
        ),
        "F1-Score": f1_score(
            y, y_pred,
            pos_label=1,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            (y == 1).astype(int),
            y_prob
        )
    }

    # Print metrics
    # print("\n" + "=" * 60)
    # print(f"{model_name} - {dataset_name}")
    # print("=" * 60)

    # print("\nPerformance Metrics:")
    # for metric, value in metrics.items():
    #     if metric not in ["Model", "Dataset"]:
    #         print(f"{metric:<15}: {value:.4f}")

    # # Confusion Matrix
    # cm = confusion_matrix(
    #     y,
    #     y_pred,
    #     labels=["repaid", "defaulted"]
    # )

    # print("\nConfusion Matrix:")
    # print("                  Predicted")
    # print("                  repaid  defaulted")
    # print(f"Actual repaid     {cm[0][0]:>6}  {cm[0][1]:>9}")
    # print(f"Actual defaulted  {cm[1][0]:>6}  {cm[1][1]:>9}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y,
        y_pred,
        labels=[0, 1],
        target_names=["Repaid", "Defaulted"],
        zero_division=0
    ))

    return metrics

df = pd.read_csv(DATA_PATH)
df = df[df['actual_outcome'] != 'ongoing'].drop(columns=["applicant_id", "days_to_default", "rule_based_decision"])
df['actual_outcome'] = df['actual_outcome'].map({'defaulted': 1, 'repaid': 0})
a = df["stated_monthly_income"]
b = df["documented_monthly_income"]
# df["perc_diff"] = (a - b).abs() / ((a + b) / 2)
# # df["stated_income_to_loan"] = a / df["loan_amount"]
# df["documented_income_to_loan"] = b / df["loan_amount"]
# df["debt_to_income"] = df["monthly_withdrawals"] / df["monthly_deposits"]
# df = df.drop(columns=["stated_monthly_income", "documented_monthly_income", "loan_amount", "monthly_withdrawals", "monthly_deposits", "num_documents_submitted","rule_based_score"])
# Separate features and target
label_col = "actual_outcome"
X = df.drop(columns=[label_col])
y = df[label_col]

# Identify column types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    )),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Logistic Regression pipeline
lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=5000,
        # class_weight="balanced"
    ))
])

# Random Forest pipeline
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=5,
        # class_weight="balanced",
        random_state=42
    ))
])

# XGBoost pipeline
xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=3,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    ))
])

# Neural Network pipeline
nn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(32,16),
        activation="relu",
        solver="adam",
        # alpha=0.01,
        # learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=30,
        random_state=42
    ))
])

from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)
sample_weights = np.where(y_train == 1, 1.0, 1.0)
# Cross-validation on training data
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

lr_scores = cross_val_score(
    lr_model, X_train, y_train,
    cv=cv, scoring="roc_auc"
)

rf_scores = cross_val_score(
    rf_model, X_train, y_train,
    cv=cv, scoring="roc_auc"
)

xgb_scores = cross_val_score(
    xgb_model, X_train, y_train,
    cv=cv, scoring="roc_auc"
)

nn_scores = cross_val_score(
    nn_model, X_train, y_train,
    cv=cv, scoring="roc_auc"
)

print("Logistic Regression:", lr_scores.mean())
print("Random Forest:", rf_scores.mean())
print("XGBoost:", xgb_scores.mean())
print("Neural Net:", nn_scores.mean())

# Fit final models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train, classifier__sample_weight=sample_weights)

results = []
# Logistic Regression
results.append(
    evaluate_model(
        lr_model, X_train, y_train,
        "Logistic Regression", "Training"
    )
)

results.append(
    evaluate_model(
        lr_model, X_test, y_test,
        "Logistic Regression", "Testing"
    )
)

# Random Forest
results.append(
    evaluate_model(
        rf_model, X_train, y_train,
        "Random Forest", "Training"
    )
)

results.append(
    evaluate_model(
        rf_model, X_test, y_test,
        "Random Forest", "Testing"
    )
)

# XGBoost
results.append(
    evaluate_model(
        xgb_model, X_train, y_train,
        "XGBoost", "Training"
    )
)

results.append(
    evaluate_model(
        xgb_model, X_test, y_test,
        "XGBoost", "Testing"
    )
)

# Neural Network
results.append(
    evaluate_model(
        nn_model, X_train, y_train,
        "Neural Network", "Training"
    )
)

results.append(
    evaluate_model(
        nn_model, X_test, y_test,
        "Neural Network", "Testing"
    )
)

results_df = pd.DataFrame(results)

print("\nFinal Model Comparison:")
print(results_df.round(4).to_string(index=False))