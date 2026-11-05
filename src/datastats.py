import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_applications.csv"

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

# Missing data plot
missing = df.isnull().mean().sort_values(ascending=False) * 100

print("Missing Values (%):")
print(missing)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=missing.values, y=missing.index, color="steelblue")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=5)

plt.title("Percentage of Missing Values by Column")
plt.xlabel("Missing Values (%)")
plt.ylabel("Column")
plt.tight_layout()
plt.show()

# Label distribution
label_col = "actual_outcome"

print("\nLabel Distribution:")
print(df[label_col].value_counts())

plt.figure(figsize=(8, 5))
ax = sns.countplot(
    data=df,
    x=label_col,
    order=df[label_col].value_counts().index,
    hue=label_col,
    palette="Set2",
    legend=False
)

for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)

plt.title("Distribution of Actual Loan Outcomes")
plt.xlabel("Actual Outcome")
plt.ylabel("Number of Applications")
plt.tight_layout()
plt.show()

# Feature correlations
label_num_col = "actual_outcome_num"
df[label_num_col] = (df[label_col] == "defaulted").astype(int)

# Optional engineered features
a = df["stated_monthly_income"]
b = df["documented_monthly_income"]
df["perc_diff"] = (a - b).abs() / b.abs()
df["stated_income_to_loan"] = a / df["loan_amount"]
df["documented_income_to_loan"] = b / df["loan_amount"]
df["debt_to_income"] = df["monthly_withdrawals"] / df["monthly_deposits"]

correlations = (
    df.corr(numeric_only=True)[label_num_col]
    .drop(label_num_col).dropna().sort_values(ascending=False)
)

print("Correlations with Defaulting:")
print(correlations)

engineered_features = [
    "perc_diff",
    "stated_income_to_loan",
    "documented_income_to_loan",
    "debt_to_income"
]

# Differentiate via colours
colors = [
    "darkorange" if feature in engineered_features else "steelblue"
    for feature in correlations.index
]

plt.figure(figsize=(11, 7))

bars = plt.barh(
    correlations.index,
    correlations.values,
    color=colors
)

for bar in bars:
    value = bar.get_width()

    plt.text(
        value + (0.01 if value >= 0 else -0.01),
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        ha="left" if value >= 0 else "right",
        fontsize=9
    )

plt.axvline(0, color="black", linewidth=1)

plt.title("Feature Correlations with Loan Default")
plt.xlabel("Pearson Correlation")
plt.ylabel("Feature")

# Legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="steelblue", label="Original Features"),
    Patch(facecolor="darkorange", label="Engineered Features")
]

plt.legend(handles=legend_elements, loc="best")

plt.tight_layout()
plt.show()
