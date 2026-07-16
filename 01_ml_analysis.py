"""
Greenside Villas — Digital Marketing & Booking Analysis
Portfolio Section 5: Machine Learning

HONEST SCOPE NOTE
This dataset has ~77 weekly rows and only 9 weeks with a booking. That is
NOT enough data for real deep learning (a TensorFlow/PyTorch model would
memorize the training set instantly and tell us nothing). This script uses
scikit-learn instead: logistic regression and random forest for a
classification task, linear/ridge regression for a forecasting task, and
one small neural network (MLPClassifier) included specifically to show the
technique — with an explicit discussion of why it doesn't outperform the
simpler models here. Every result below is reported with train/test
discipline and the small-sample caveat stated up front, not buried at
the end.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, r2_score
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130
NAVY, TEAL, CORAL, GOLD = "#1F4E78", "#2E9E8E", "#E4572E", "#D9A441"

# ---------------------------------------------------------------------------
# 1. LOAD + BUILD WEEKLY FEATURE SET
# ---------------------------------------------------------------------------
fb = pd.read_csv("facebook_weekly.csv", parse_dates=["week_date"])
ig = pd.read_csv("instagram_weekly.csv", parse_dates=["week_date"])
web = pd.read_csv("website_weekly.csv", parse_dates=["week_date"])
ab = pd.read_csv("airbnb_weekly.csv", parse_dates=["week_date"])

df = fb[["week_date", "reach"]].rename(columns={"reach": "fb_reach"})
df = df.merge(ig[["week_date", "reach"]].rename(columns={"reach": "ig_reach"}), on="week_date", how="left")
df = df.merge(web[["week_date", "visitors"]].rename(columns={"visitors": "site_visitors"}), on="week_date", how="left")
df = df.merge(ab[["week_date", "views", "inquiries", "bookings", "revenue"]]
              .rename(columns={"views": "airbnb_views", "inquiries": "airbnb_inquiries",
                                "bookings": "airbnb_bookings", "revenue": "airbnb_revenue"}),
              on="week_date", how="left")
df = df.sort_values("week_date").reset_index(drop=True)
df = df[df["airbnb_bookings"].notna()].reset_index(drop=True)  # drop future placeholder weeks

# Lag features: use LAST week's activity to predict THIS week's booking outcome
# (avoids same-week leakage between inquiries and bookings)
for col in ["fb_reach", "ig_reach", "site_visitors", "airbnb_views", "airbnb_inquiries"]:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_roll4"] = df[col].rolling(4, min_periods=1).mean().shift(1)

df["booked_this_week"] = (df["airbnb_bookings"] > 0).astype(int)
df = df.dropna().reset_index(drop=True)

print(f"Total usable weeks after lag/rolling features: {len(df)}")
print(f"Weeks with a booking: {df['booked_this_week'].sum()} "
      f"({df['booked_this_week'].mean()*100:.1f}%)")

feature_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_roll4")]

# ---------------------------------------------------------------------------
# 2. TASK A — CLASSIFY: will THIS week produce a booking?
# Time-based split (not random) — train on first ~75%, test on last ~25%,
# because shuffling weeks would leak future information into training.
# ---------------------------------------------------------------------------
split_idx = int(len(df) * 0.75)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
print(f"\nTrain weeks: {len(train)} (bookings: {train['booked_this_week'].sum()})")
print(f"Test weeks:  {len(test)} (bookings: {test['booked_this_week'].sum()})")

X_train, y_train = train[feature_cols], train["booked_this_week"]
X_test, y_test = test[feature_cols], test["booked_this_week"]

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42, class_weight="balanced"),
    "Neural Net (MLPClassifier)": MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    })
    print(f"\n{name}")
    print(confusion_matrix(y_test, preds))

results_df = pd.DataFrame(results)
print("\nClassification results:")
print(results_df.to_string(index=False))

# Baseline: "always predict no booking"
baseline_acc = (y_test == 0).mean()
print(f"\nBaseline ('always predict no booking') accuracy: {baseline_acc:.3f}")

# Feature importance from Random Forest
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nRandom Forest feature importances:")
print(importances.to_string())

# ---------------------------------------------------------------------------
# CHART: Confusion matrix for the best-performing model (by F1)
# ---------------------------------------------------------------------------
best_name = results_df.sort_values("f1", ascending=False).iloc[0]["model"]
best_model = models[best_name]
cm = confusion_matrix(y_test, best_model.predict(X_test_s))
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No booking", "Booking"], yticklabels=["No booking", "Booking"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name}\n(test set, {len(test)} weeks)", fontweight="bold")
plt.tight_layout()
plt.savefig("charts/01_confusion_matrix.png")
plt.close()

# CHART: Feature importance
fig, ax = plt.subplots(figsize=(8, 5))
importances.sort_values().plot(kind="barh", color=NAVY, ax=ax)
ax.set_title("Random Forest Feature Importance — Predicting Weekly Booking", fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("charts/02_feature_importance.png")
plt.close()

# ---------------------------------------------------------------------------
# 3. TASK B — FORECAST: next month's Airbnb revenue
# ---------------------------------------------------------------------------
df["month"] = df["week_date"].dt.to_period("M")
monthly = df.groupby("month").agg(
    fb_reach=("fb_reach", "sum"), ig_reach=("ig_reach", "sum"),
    site_visitors=("site_visitors", "sum"), airbnb_views=("airbnb_views", "sum"),
    airbnb_inquiries=("airbnb_inquiries", "sum"), airbnb_bookings=("airbnb_bookings", "sum"),
    airbnb_revenue=("airbnb_revenue", "sum"),
).reset_index()

# Predict THIS month's revenue from LAST month's features (real forecasting setup)
for col in ["fb_reach", "ig_reach", "site_visitors", "airbnb_views", "airbnb_inquiries", "airbnb_revenue"]:
    monthly[f"{col}_lag1"] = monthly[col].shift(1)
monthly_ml = monthly.dropna().reset_index(drop=True)

m_split = int(len(monthly_ml) * 0.75)
m_train, m_test = monthly_ml.iloc[:m_split], monthly_ml.iloc[m_split:]
m_features = [c for c in monthly_ml.columns if c.endswith("_lag1")]

Xm_train, ym_train = m_train[m_features], m_train["airbnb_revenue"]
Xm_test, ym_test = m_test[m_features], m_test["airbnb_revenue"]

ridge = Ridge(alpha=1.0).fit(Xm_train, ym_train)
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42).fit(Xm_train, ym_train)

ridge_pred = ridge.predict(Xm_test)
rf_pred = rf_reg.predict(Xm_test)
baseline_pred = np.full(len(ym_test), ym_train.mean())  # predict train-set average every time

print(f"\nMonthly forecast test set: {len(m_test)} months")
print(f"Ridge Regression  MAE: ${mean_absolute_error(ym_test, ridge_pred):.0f}")
print(f"Random Forest     MAE: ${mean_absolute_error(ym_test, rf_pred):.0f}")
print(f"Baseline (mean)   MAE: ${mean_absolute_error(ym_test, baseline_pred):.0f}")

# CHART: Actual vs predicted revenue
fig, ax = plt.subplots(figsize=(9, 5))
x_labels = m_test["month"].astype(str)
ax.plot(x_labels, ym_test.values, marker="o", color=NAVY, linewidth=2, label="Actual revenue")
ax.plot(x_labels, ridge_pred, marker="s", color=CORAL, linestyle="--", label="Ridge prediction")
ax.plot(x_labels, baseline_pred, marker="x", color="#999999", linestyle=":", label="Baseline (mean)")
ax.set_title("Monthly Revenue Forecast — Actual vs Predicted (test months)", fontweight="bold")
ax.set_ylabel("Revenue ($)")
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("charts/03_revenue_forecast.png")
plt.close()

print("\nAll charts saved to ./charts/")
