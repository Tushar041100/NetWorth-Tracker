import pandas as pd
import numpy as np
import random
 
# Set random seed for reproducibility
np.random.seed(42)
 
# Constants
NUM_USERS = 1000
RISK_LEVELS = ['Low', 'Medium', 'High']
GOALS = ['Retirement', 'House', 'Education', 'Wealth Creation']
EXPERIENCE = ['Beginner', 'Intermediate', 'Expert']
 
def generate_user():
    age = np.random.randint(21, 60)
    income = np.random.randint(20000, 300000)
    expenses = np.random.randint(int(0.3 * income), int(0.8 * income))
    risk = random.choices(RISK_LEVELS, weights=[0.3, 0.5, 0.2])[0]
    goal = random.choice(GOALS)
    horizon = np.random.randint(1, 30)
    experience = random.choices(EXPERIENCE, weights=[0.5, 0.3, 0.2])[0]
    
    return {
        "age": age,
        "income_per_month": income,
        "monthly_expenses": expenses,
        "risk_tolerance": risk,
        "financial_goal": goal,
        "goal_horizon_years": horizon,
        "investment_experience": experience
    }
 
def get_asset_allocation(risk, horizon, income, expenses):
    equity, debt, gold, real_estate = 0, 0, 0, 0
    savings_ratio = (income - expenses) / income
 
    # Base allocation by risk
    if risk == 'Low':
        equity = np.random.uniform(10, 25)
        debt = np.random.uniform(50, 70)
        gold = np.random.uniform(5, 10)
    elif risk == 'Medium':
        equity = np.random.uniform(40, 60)
        debt = np.random.uniform(20, 40)
        gold = np.random.uniform(5, 10)
    else:  # High risk
        equity = np.random.uniform(65, 85)
        debt = np.random.uniform(5, 20)
        gold = np.random.uniform(5, 10)
 
    # Adjust based on horizon
    if horizon < 3:
        equity *= 0.7
        debt *= 1.1
    elif horizon > 10:
        equity *= 1.1
        debt *= 0.9
 
    # Adjust gold slightly for inflation hedge
    if horizon >= 10:
        gold += 2
 
    # Real estate allocation for high-income users
    if income > 150000 and savings_ratio > 0.3:
        real_estate = np.random.uniform(5, 15)
 
    # Normalize to 100%
    total = equity + debt + gold + real_estate
    equity = (equity / total) * 100
    debt = (debt / total) * 100
    gold = (gold / total) * 100
    real_estate = (real_estate / total) * 100
 
    return equity, debt, gold, real_estate
 
# Generate dataset
data = []
for _ in range(NUM_USERS):
    user = generate_user()
    equity, debt, gold, real_estate = get_asset_allocation(
        user['risk_tolerance'], user['goal_horizon_years'],
        user['income_per_month'], user['monthly_expenses']
    )
    user.update({
        "equity_percent": round(equity, 2),
        "debt_percent": round(debt, 2),
        "gold_percent": round(gold, 2),
        "real_estate_percent": round(real_estate, 2)
    })
    data.append(user)
 
# Create DataFrame
df = pd.DataFrame(data)
 
# Save to CSV
df.to_csv("synthetic_asset_allocation_dataset.csv", index=False)
print("Dataset generated and saved as 'synthetic_asset_allocation_dataset.csv'")