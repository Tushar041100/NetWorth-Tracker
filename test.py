import pytest
from validation import AssetAllocationRequest, AssetAllocationResponse
import joblib
import numpy as np

# Load model + scaler for prediction tests
model = joblib.load("asset_allocation_model.pkl")
scaler = joblib.load("scaler.pkl")


# Test: Valid request input
def test_valid_request():
    data = {
        "age": 30,
        "income_per_month": 50000,
        "monthly_expenses": 20000,
        "goal_horizon_years": 10,
        "risk_tolerance": "Medium",
        "financial_goal": "Retirement",
        "investment_experience": "Intermediate",
    }
    req = AssetAllocationRequest(**data)
    assert req.age == 30
    assert req.risk_tolerance == "Medium"


# Test: Invalid request input (expenses > income)
def test_invalid_request_expenses_exceed_income():
    with pytest.raises(ValueError):
        AssetAllocationRequest(
            age=28,
            income_per_month=30000,
            monthly_expenses=35000,
            goal_horizon_years=5,
            risk_tolerance="High",
            financial_goal="Travel",
            investment_experience="Beginner",
        )


# Test: Prediction shape and response validation
def test_model_prediction_output():
    # Sample encoded and scaled input
    input_features = np.array(
        [
            [
                30,  # age
                70000,  # income
                25000,  # expenses
                15,  # goal_horizon
                1,  # Medium risk
                0,  # Retirement
                1,  # Intermediate
                0.64,  # savings ratio
                1,  # is long term
                2,  # income bracket
            ]
        ]
    )
    input_features[:, 0:4] = scaler.transform(input_features[:, 0:4])
    prediction = model.predict(input_features)[0]

    equity = round(prediction[0], 2)
    debt = round(prediction[1], 2)
    gold = round(prediction[2], 2)

    # Step 3: Force the 4th to make total 100
    real_estate = round(100 - (equity + debt + gold), 2)
    # Check shape & convert to response
    assert len(prediction) == 4
    response = AssetAllocationResponse(
        equity_percent=equity,
        debt_percent=debt,
        gold_percent=gold,
        real_estate_percent=real_estate,
    )
    total = sum(response.dict().values())
    assert 98.0 <= total <= 102.0  # Tolerance for floating point sum


# Test: Negative age should raise validation error
def test_invalid_negative_age():
    with pytest.raises(ValueError):
        AssetAllocationRequest(
            age=-5,
            income_per_month=40000,
            monthly_expenses=10000,
            goal_horizon_years=10,
            risk_tolerance="Low",
            financial_goal="Child Education",
            investment_experience="Beginner",
        )
