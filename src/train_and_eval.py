import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    TunedThresholdClassifierCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.calibration import CalibratedClassifierCV

from sklearn.base import clone

import shap
import matplotlib.pyplot as plt

class Data:
    def __init__(self, df):
        X, y = self.engineer_and_split_data(df)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

    def engineer_and_split_data(self, df):
        df = df[df['actual_outcome'] != 'ongoing'].drop(columns=["applicant_id", "days_to_default", "rule_based_decision"])
        df['actual_outcome'] = df['actual_outcome'].map({'defaulted': 1, 'repaid': 0})

        # Engineered features
        a = df["stated_monthly_income"]
        b = df["documented_monthly_income"]

        df["perc_diff"] = (
            (a - b).abs() / ((a + b) / 2).replace(0, np.nan)
        )

        df["documented_income_to_loan"] = (
            b / df["loan_amount"].replace(0, np.nan)
        )

        df["debt_to_income"] = (
            df["monthly_withdrawals"] /
            df["monthly_deposits"].replace(0, np.nan)
        )

        label_col = "actual_outcome"
        X = df.drop(columns=[label_col])
        y = df[label_col]
        return X, y


class ClassifierTrainer:
    def __init__(self, model, data, name=None, threshold=0.5):
        self.model = model
        self.model_ = clone(self.model)
        self.data = data
        self.name = name if name is not None else type(model).__name__
        self.threshold = threshold
        
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
            "selector": SelectKBest(score_func=f_classif, k=int(len(self.data.X_train.columns)*2/3)),
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

        numeric_features = self.data.X_train.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = self.data.X_train.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        full_pipeline = [("preprocessor", preprocessor)]

        if given["selector"] != EXCLUDE_VAL:
            full_pipeline.append(("selector", given["selector"]))

        full_pipeline.append(("classifier", clone(self.model)))
        self.model_ = Pipeline(full_pipeline)

    def train_model(
        self,
        n_splits=4,
        param_dist=None,
        tune_threshold=False,
        calibrate=False,
        calibration_method="sigmoid",
        metric="f1",
        **fit_kwargs
    ):
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        if param_dist is not None:
            search = RandomizedSearchCV(
                estimator=self.model_,
                param_distributions=param_dist,
                scoring=metric,
                cv=cv,
                n_jobs=1,
                random_state=42,
                verbose=2
            )
            search.fit(self.data.X_train, self.data.y_train, **fit_kwargs)
            self.model_ = search.best_estimator_

        if calibrate:
            self.model_ = CalibratedClassifierCV(
                estimator=self.model_,
                method=calibration_method,
                cv=cv,
                ensemble=False
        )

        if tune_threshold:
            self.threshold = None
            self.model_ = TunedThresholdClassifierCV(
                estimator=self.model_,
                scoring=metric,
                cv=cv
            )

        self.model_.fit(self.data.X_train, self.data.y_train, **fit_kwargs)

    def get_model(self):
        return self.model_

    def evaluate_model(self, splits=None):
        splits = splits or ["Testing"]
        metrics = []

        for split in splits:
            X_split = self.data.X_train if split == "Training" else self.data.X_test
            y_split = self.data.y_train if split == "Training" else self.data.y_test

            # Make predictions
            to_eval = self.get_model()
            y_pred = to_eval.predict(X_split)

            # Get probability of default
            defaulted_index = list(to_eval.classes_).index(1)
            y_prob = to_eval.predict_proba(X_split)[:, defaulted_index]
            if self.threshold is None:
                y_pred = to_eval.predict(X_split)
            else:
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
                "ROC-AUC": roc_auc_score(y_split, y_prob),
                "PR-AUC": average_precision_score(y_split, y_prob),
                "MCC": matthews_corrcoef(y_split, y_pred)
            })

            # Print stats
            print("\n" + "=" * 60)
            print(f"{self.name} - {split}")
            print("=" * 60)

            # Confusion Matrix
            cm = confusion_matrix(
                y_split,
                y_pred,
                labels=[0, 1]
            )

            print("\nConfusion Matrix:")
            print("                  Predicted")
            print("                  repaid  defaulted")
            print(f"Actual repaid     {cm[0][0]:>6}  {cm[0][1]:>9}")
            print(f"Actual defaulted  {cm[1][0]:>6}  {cm[1][1]:>9}")

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

    def explain_shap(
        self,
        X_background=None,
        X_explain=None,
        background_size=100,
        explain_size=200,
        random_state=42
    ):
        """
        Generate SHAP explanations for the complete pipeline,
        supporting numerical and categorical input features.
        """

        if X_background is None:
            X_background = self.data.X_train.copy()

        if X_explain is None:
            X_explain = self.data.X_test.copy()

        # Sample datasets
        if len(X_background) > background_size:
            X_background = X_background.sample(
                n=background_size,
                random_state=random_state
            )

        if len(X_explain) > explain_size:
            X_explain = X_explain.sample(
                n=explain_size,
                random_state=random_state
            )

        # Preserve original feature names and column order
        self.X_shap_ = X_explain.copy()

        # Convert categorical columns into numerical codes
        X_background_encoded = X_background.copy()
        X_explain_encoded = X_explain.copy()

        self.shap_categories_ = {}

        for col in X_background.columns:
            if (
                X_background[col].dtype == "object"
                or isinstance(X_background[col].dtype, pd.CategoricalDtype)
                or X_background[col].dtype.name in ["str", "bool"]
            ):

                # Create consistent category mapping
                categories = pd.Index(
                    X_background[col].dropna().unique()
                )

                self.shap_categories_[col] = categories

                X_background_encoded[col] = pd.Categorical(
                    X_background[col],
                    categories=categories
                ).codes.astype(float)

                X_explain_encoded[col] = pd.Categorical(
                    X_explain[col],
                    categories=categories
                ).codes.astype(float)

        # Handle any remaining non-numeric columns
        X_background_encoded = X_background_encoded.astype(float)
        X_explain_encoded = X_explain_encoded.astype(float)

        # Prediction function
        def predict_default(X):

            X = pd.DataFrame(
                X,
                columns=self.X_shap_.columns
            )

            # Restore original categorical values
            for col, categories in self.shap_categories_.items():

                codes = X[col].to_numpy().astype(int)

                values = [
                    categories[code] if 0 <= code < len(categories)
                    else np.nan
                    for code in codes
                ]

                X[col] = values

            probabilities = self.model_.predict_proba(X)

            default_index = list(self.model_.classes_).index(1)

            return probabilities[:, default_index]

        # SHAP explainer
        self.shap_explainer_ = shap.Explainer(
            predict_default,
            X_background_encoded,
            algorithm="permutation"
        )

        self.shap_values_ = self.shap_explainer_(
            X_explain_encoded,
            max_evals=10 * X_explain.shape[1] + 1
        )

        # Restore original feature names
        self.shap_values_.feature_names = list(self.X_shap_.columns)

        print("SHAP explanations generated successfully.")
        print(f"Observations: {len(X_explain)}")
        print(f"Features: {X_explain.shape[1]}")

        return self.shap_values_

    def plot_shap_summary(self, max_display=20):
        """
        Global SHAP summary (beeswarm) plot.
        Shows feature importance and impact on predictions.
        """

        if not hasattr(self, "shap_values_"):
            raise ValueError("Run explain_shap() first.")

        shap.plots.beeswarm(
            self.shap_values_,
            max_display=max_display,
            show=False
        )

        plt.title(f"{self.name} - SHAP Summary")
        plt.tight_layout()
        plt.show()

    def plot_shap_bar(self, max_display=20):
        """
        Global feature importance bar plot.
        Features are ranked by mean absolute SHAP value.
        """

        if not hasattr(self, "shap_values_"):
            raise ValueError("Run explain_shap() first.")

        shap.plots.bar(
            self.shap_values_,
            max_display=max_display,
            show=False
        )

        plt.title(f"{self.name} - SHAP Feature Importance")
        plt.tight_layout()
        plt.show()

    def plot_shap_waterfall(self, observation=0, max_display=20):
        """
        Local SHAP waterfall plot.
        Explains an individual prediction.
        """

        if not hasattr(self, "shap_values_"):
            raise ValueError("Run explain_shap() first.")

        shap.plots.waterfall(
            self.shap_values_[observation],
            max_display=max_display,
            show=False
        )

        plt.title(f"{self.name} - SHAP Waterfall")
        plt.tight_layout()
        plt.show()

    def plot_shap_dependence(self, feature, max_display=20):
        """
        SHAP dependence/scatter plot.
        Shows how a feature affects model predictions.
        """

        if not hasattr(self, "shap_values_"):
            raise ValueError("Run explain_shap() first.")

        shap.plots.scatter(
            self.shap_values_[:, feature],
            color=self.shap_values_,
            show=False
        )

        plt.title(f"{self.name} - SHAP Dependence: {feature}")
        plt.tight_layout()
        plt.show()

def fairness_analysis(
    data,
    model,
    baseline_decisions,
    employment_col="employment_status",
    threshold=0.5
):

    X_test = data.X_test
    y_test = data.y_test

    # Align baseline decisions with test observations
    baseline = baseline_decisions.loc[X_test.index]

    baseline_approved = (
        baseline == "approved"
    ).astype(int)

    # Random Forest predictions
    probabilities = model.predict_proba(X_test)

    default_index = list(model.classes_).index(1)

    default_probabilities = probabilities[:, default_index]

    rf_approved = (
        default_probabilities < threshold
    ).astype(int)

    # Construct evaluation dataframe
    results = pd.DataFrame({
        "employment_status": X_test[employment_col],
        "actual_default": y_test,
        "baseline_approved": baseline_approved,
        "model_approved": rf_approved
    }, index=X_test.index)

    def calculate_metrics(group, approval_column):

        approved = group[approval_column] == 1
        repaid = group["actual_default"] == 0
        defaulted = group["actual_default"] == 1

        total = len(group)
        approved_count = approved.sum()

        return {
            "Applicants": total,

            "Approval Rate": (
                approved.mean()
            ),

            "Actual Default Rate": (
                defaulted.mean()
            ),

            "Default Rate Among Approved": (
                group.loc[approved, "actual_default"].mean()
                if approved_count > 0 else np.nan
            ),

            "Approval Rate Among Repaid": (
                (approved & repaid).sum() / repaid.sum()
                if repaid.sum() > 0 else np.nan
            )
        }

    fairness_results = []

    for employment_status, group in results.groupby(
        "employment_status",
        dropna=False
    ):

        for model_name, approval_column in [
            ("Baseline", "baseline_approved"),
            ("Random Forest", "model_approved")
        ]:

            metrics = calculate_metrics(
                group,
                approval_column
            )

            metrics["Employment Status"] = employment_status
            metrics["Model"] = model_name

            fairness_results.append(metrics)

    fairness_df = pd.DataFrame(fairness_results)

    fairness_df = fairness_df[
        [
            "Model",
            "Employment Status",
            "Applicants",
            "Approval Rate",
            "Actual Default Rate",
            "Default Rate Among Approved",
            "Approval Rate Among Repaid"
        ]
    ]

    percentage_columns = [
        "Approval Rate",
        "Actual Default Rate",
        "Default Rate Among Approved",
        "Approval Rate Among Repaid"
    ]

    fairness_df[percentage_columns] *= 100

    print("\n" + "=" * 80)
    print("FAIRNESS ANALYSIS: EMPLOYMENT STATUS")
    print("=" * 80)

    print(
        fairness_df.round(2).to_string(index=False)
    )

    return fairness_df