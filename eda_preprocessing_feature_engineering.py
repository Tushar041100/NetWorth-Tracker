import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('synthetic_asset_allocation_dataset.csv')
 
# Basic overview
print(df.head())
print(df.info())
print(df.describe())
 
# Check for missing values
print("Missing values:\n", df.isnull().sum())
 
# Categorical distributions
for col in ['risk_tolerance', 'financial_goal', 'investment_experience']:
    print(f"\nValue counts for {col}:\n", df[col].value_counts())
 
# Visualize numeric distributions
num_cols = ['age', 'income_per_month', 'monthly_expenses', 'goal_horizon_years']
df[num_cols].hist(bins=20, figsize=(12, 8))
plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()
plt.show()
 
# Correlation heatmap for numerical features
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
 
# Boxplots for asset allocation vs risk
for asset in ['equity_percent', 'debt_percent', 'gold_percent', 'real_estate_percent']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='risk_tolerance', y=asset, data=df)
    plt.title(f"{asset} vs Risk Tolerance")
plt.show()

df = pd.read_csv('synthetic_asset_allocation_dataset.csv')
 
# Label Encoding for categoricals
label_encoders = {}
for col in ['risk_tolerance', 'financial_goal', 'investment_experience']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
 
# Feature and target split
X = df.drop(columns=[
    'equity_percent', 'debt_percent', 'gold_percent', 'real_estate_percent'
])
y = df[['equity_percent', 'debt_percent', 'gold_percent', 'real_estate_percent']]
 
# Normalize income, expenses, age, horizon
scaler = StandardScaler()
X[['age', 'income_per_month', 'monthly_expenses', 'goal_horizon_years']] = scaler.fit_transform(
    X[['age', 'income_per_month', 'monthly_expenses', 'goal_horizon_years']]
)
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
print("Preprocessing complete.")

# Add savings rate
df['savings_ratio'] = (df['income_per_month'] - df['monthly_expenses']) / df['income_per_month']
 
# IsLongTermGoal
df['is_long_term_goal'] = (df['goal_horizon_years'] > 10).astype(int)
 
# Income Bracket (Low/Medium/High)
df['income_bracket'] = pd.cut(
    df['income_per_month'],
    bins=[0, 50000, 150000, np.inf],
    labels=['Low', 'Medium', 'High']
)
 
# Encode new feature
df['income_bracket'] = LabelEncoder().fit_transform(df['income_bracket'])
 
# Update X features
X = df.drop(columns=[
    'equity_percent', 'debt_percent', 'gold_percent', 'real_estate_percent'
])
y = df[['equity_percent', 'debt_percent', 'gold_percent', 'real_estate_percent']]