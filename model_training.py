import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("synthetic_asset_allocation_dataset.csv")

# Feature Engineering
df["savings_ratio"] = (df["income_per_month"] - df["monthly_expenses"]) / df["income_per_month"]
df["is_long_term_goal"] = (df["goal_horizon_years"] > 10).astype(int)
df["income_bracket"] = pd.cut(
    df["income_per_month"],
    bins=[0, 50000, 150000, float("inf")],
    labels=["Low", "Medium", "High"],
)
df["income_bracket"] = LabelEncoder().fit_transform(df["income_bracket"])

# Encode categoricals
for col in ["risk_tolerance", "financial_goal", "investment_experience"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# Select features and targets
X = df[
    [
        "age",
        "income_per_month",
        "monthly_expenses",
        "goal_horizon_years",
        "risk_tolerance",
        "financial_goal",
        "investment_experience",
        "savings_ratio",
        "is_long_term_goal",
        "income_bracket",
    ]
]
y = df[["equity_percent", "debt_percent", "gold_percent", "real_estate_percent"]]

# Normalize numerical features
scaler = StandardScaler()
X[["age", "income_per_month", "monthly_expenses", "goal_horizon_years"]] = scaler.fit_transform(
    X[["age", "income_per_month", "monthly_expenses", "goal_horizon_years"]]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model and scaler
joblib.dump(model, "asset_allocation_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")
