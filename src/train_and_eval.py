import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
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

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV, TunedThresholdClassifierCV
from scipy.stats import randint, uniform, loguniform
from sklearn.metrics import fbeta_score

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_applications.csv"

MODEL_THRESHOLD = 0.5

class ClassifierTrainer:
    def __init__(self, model, X, y, name=None, threshold=0.5):
        self.model = model
        self.name = name if name is not None else type(model).__name__
        self.threshold = threshold
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
    def build_pipeline(
        self,
        num_imputer=None,
        cat_imputer=None,
        scaler=None,
        selector=None,
        encoder=None
    ):
        EXCLUDE_VAL = "passthrough"

        defaults = {
            "num_imputer": SimpleImputer(strategy="median", add_indicator=True),
            "cat_imputer": SimpleImputer(strategy="constant", fill_value="missing"),
            "scaler": StandardScaler(),
            "selector": VarianceThreshold(threshold=0.0),
            "encoder": OneHotEncoder(handle_unknown="ignore")
        }

        given = {
            "num_imputer": num_imputer,
            "cat_imputer": cat_imputer,
            "scaler": scaler,
            "selector": selector,
            "encoder": encoder
        }

        for step, val in given.items():
            if val is None:
                given[step] = defaults[step]

        def build_subpipeline(steps):
            pipe = [
                (step, given[step])
                for step in steps
                if given[step] != EXCLUDE_VAL
            ]

            return Pipeline(pipe) if pipe else "passthrough"

        numeric_pipeline = build_subpipeline(["num_imputer", "scaler"])
        categorical_pipeline = build_subpipeline(["cat_imputer", "encoder"])

        numeric_features = self.X_train.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = self.X_train.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        full_pipeline = [("preprocessor", preprocessor)]

        if given["selector"] != EXCLUDE_VAL:
            full_pipeline.append(("selector", given["selector"]))

        full_pipeline.append(("classifier", self.model))
        self.model = Pipeline(full_pipeline)

    def train_model(
        self,
        n_splits=5,
        param_dist=None,
        tune_threshold=False,
        metric="f1",
        **fit_kwargs
    ):
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        if param_dist is not None:
            self.model = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_dist,
                scoring=metric,
                cv=cv,
                n_jobs=1,
                random_state=42
            )

        if tune_threshold:
            self.model = TunedThresholdClassifierCV(
                estimator=self.model,
                scoring=metric,
                cv=cv
            )

        self.model.fit(self.X_train, self.y_train, **fit_kwargs)

    def get_model(self):
        model = self.model

        while True:
            if hasattr(model, "best_estimator_"):
                model = model.best_estimator_
            elif hasattr(model, "estimator_"):
                model = model.estimator_
            else:
                break

        return model

    def evaluate_model(self, splits=None):
        splits = splits or ["Testing"]
        metrics = []

        for split in splits:
            X_split = self.X_train if split == "Training" else self.X_test
            y_split = self.y_train if split == "Training" else self.y_test

            # Make predictions
            to_eval = self.get_model()
            y_pred = to_eval.predict(X_split)

            # Get probability of default
            defaulted_index = list(to_eval.classes_).index(1)
            y_prob = to_eval.predict_proba(X_split)[:, defaulted_index]
            y_pred = (y_prob >= self.threshold).astype(int)
            # m = {1:"defaulted", 0:"repaid"}
            # y_pred = np.array([m[x] for x in y_pred])
            # Calculate metrics
            metrics.append({
                "Model": self.name,
                "Dataset": split,
                "Accuracy": accuracy_score(y_split, y_pred),
                "Precision": precision_score(
                    y_split, y_pred,
                    pos_label=1,
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_split, y_pred,
                    pos_label=1,
                    zero_division=0
                ),
                "F1-Score": f1_score(
                    y_split, y_pred,
                    pos_label=1,
                    zero_division=0
                ),
                "ROC-AUC": roc_auc_score(
                    (y_split == 1).astype(int),
                    y_prob
                )
            })

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
            #     y_split,
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
                y_split,
                y_pred,
                labels=[0, 1],
                target_names=["Repaid", "Defaulted"],
                zero_division=0
            ))

        return metrics


df = pd.read_csv(DATA_PATH)
rule_based_decision = df["rule_based_decision"]
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Logistic Regression pipeline
lr_model = ClassifierTrainer(
    LogisticRegression(max_iter=5000),
    X, y,
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
    X, y,
    threshold=MODEL_THRESHOLD
)
rf_model.build_pipeline()

# XGBoost pipeline
xgb_model = ClassifierTrainer(
    XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ),
    X, y,
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
    X, y,
    threshold=MODEL_THRESHOLD
)
nn_model.build_pipeline()

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)
sample_weights = np.where(y_train == 1, 1.0, 1.0)


def f2_scorer(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= MODEL_THRESHOLD).astype(int)
    return fbeta_score(y, y_pred, beta=2)

param_dist = {
    "classifier__n_estimators": randint(100, 401),
    "classifier__max_depth": randint(1, 4),
    "classifier__learning_rate": loguniform(0.005, 0.05),
    "classifier__min_child_weight": randint(5, 20),
    "classifier__subsample": uniform(0.6, 0.35),
    "classifier__colsample_bytree": uniform(0.5, 0.4),
    "classifier__gamma": loguniform(0.01, 5),
    "classifier__reg_alpha": loguniform(0.1, 50),
    "classifier__reg_lambda": loguniform(5, 100)
}

# Fit final models
lr_model.train_model()
rf_model.train_model()
xgb_model.train_model(param_dist=param_dist)
nn_model.train_model(classifier__sample_weight=sample_weights)

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
rule_based_test = rule_based_decision.loc[y_test.index]
y_pred_baseline = rule_based_test.map(map_base)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_baseline,
    target_names=["Repaid", "Defaulted"]
))