import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from pathlib import Path

from train_and_eval import Data, ClassifierTrainer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_applications.csv"

MODEL_THRESHOLD = 0.45

df = pd.read_csv(DATA_PATH)
rule_based_decision = df["rule_based_decision"]
data = Data(df)

# Logistic Regression pipeline
lr_model = ClassifierTrainer(
    LogisticRegression(max_iter=5000),
    data,
    threshold=MODEL_THRESHOLD
)
lr_model.build_pipeline()

# Random Forest pipeline
rf_model = ClassifierTrainer(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    ),
    data,
    threshold=MODEL_THRESHOLD
)
rf_model.build_pipeline()

# XGBoost pipeline
xgb_model = ClassifierTrainer(
    XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=7,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=5,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    ),
    data,
    threshold=MODEL_THRESHOLD
)
xgb_model.build_pipeline()

# Neural Network pipeline
nn_model = ClassifierTrainer(
    MLPClassifier(
        hidden_layer_sizes=(32,16),
        max_iter=5000,
        early_stopping=True,
        random_state=42
    ),
    data,
    threshold=MODEL_THRESHOLD
)
nn_model.build_pipeline()

# Fit final models
lr_model.train_model()
rf_model.train_model()
xgb_model.train_model()
nn_model.train_model()

results = []

results.extend(lr_model.evaluate_model(["Training", "Testing"]))
results.extend(rf_model.evaluate_model(["Training", "Testing"]))
results.extend(xgb_model.evaluate_model(["Training", "Testing"]))
results.extend(nn_model.evaluate_model(["Training", "Testing"]))

results_df = pd.DataFrame(results)

print("\nFinal Model Comparison:")
print(results_df.round(4).to_string(index=False))

# Map baseline decisions
map_base = {
    "approved": 0,
    "denied": 1,
    "flagged_for_review": 0
}

# Convert categorical values to binary
rule_based_test = rule_based_decision.loc[data.y_test.index]
y_pred_baseline = rule_based_test.map(map_base)

# Classification report
print("\nBaseline Classification Report:")
print(classification_report(
    data.y_test,
    y_pred_baseline,
    target_names=["Repaid", "Defaulted"]
))
