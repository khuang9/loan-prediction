import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from scipy.stats import randint, uniform, loguniform

from pathlib import Path

from metrics import build_f2_scorer
from train_and_eval import Data, ClassifierTrainer, fairness_analysis

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_applications.csv"

MODEL_THRESHOLD = 0.5

df = pd.read_csv(DATA_PATH)
rule_based_decision = df["rule_based_decision"]
data = Data(df)

# Random Forest pipeline
rf_model = ClassifierTrainer(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    ),
    data,
    threshold=MODEL_THRESHOLD
)
rf_model.build_pipeline()

param_dist = {
    "classifier__n_estimators": randint(200, 500),
    "classifier__max_depth": randint(2, 5),
    "classifier__min_samples_split": randint(2, 21),
    "classifier__min_samples_leaf": randint(2, 16),
    "classifier__class_weight": [None, "balanced", "balanced_subsample"],
    "classifier__criterion": ["gini", "entropy", "log_loss"]
}

# All options enabled
rf_model.train_model()

results = []

results.extend(rf_model.evaluate_model(["Training", "Testing"]))

results_df = pd.DataFrame(results)

print("\nMetrics:")
print(results_df.round(4).to_string(index=False))

# Map baseline decisions
map_base = {
    "approved": 0,
    "denied": 1,
    "flagged_for_review": 1
}

# Convert categorical values to binary
rule_based_test = rule_based_decision.loc[data.y_test.index]
y_pred_baseline = rule_based_test.map(map_base)

# Classification report
print('\n' + '='*60)
print("Baseline Classification Report:")
print('='*60)
print(classification_report(
    data.y_test,
    y_pred_baseline,
    target_names=["Repaid", "Defaulted"]
))

fairness_df = fairness_analysis(
    data=data,
    model=rf_model.get_model(),
    baseline_decisions=rule_based_decision,
    threshold=MODEL_THRESHOLD
)

print(fairness_df)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

sns.barplot(
    data=fairness_df,
    x="Employment Status",
    y="Approval Rate",
    hue="Model"
)

plt.title("Approval Rates Across Employment Groups")
plt.xlabel("Employment Status")
plt.ylabel("Approval Rate (%)")
plt.ylim(0, 100)

plt.legend(title="Model")
plt.tight_layout()
plt.show()

# Generate SHAP values
shap_values = rf_model.explain_shap(
    background_size=100,
    explain_size=200
)

# Global feature importance
rf_model.plot_shap_bar()

# Global feature impact
rf_model.plot_shap_summary()

# Explain an individual prediction
rf_model.plot_shap_waterfall(observation=0)

# Feature dependence
rf_model.plot_shap_dependence("loan_amount")


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))

tree = rf_model.get_model()
while True:
    if hasattr(tree, "best_estimator_"):
        tree = tree.best_estimator_
    elif hasattr(tree, "estimator_"):
        tree = tree.estimator_
    elif hasattr(tree, "estimator"):
            tree = tree.estimator
    else:
        break
tree = tree.named_steps['classifier']

plot_tree(
    tree.estimators_[0],
    feature_names=data.X_train.columns,
    class_names=["repaid", "defaulted"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree 1 from Random Forest")
plt.show()