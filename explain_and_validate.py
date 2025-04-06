from validation import AssetAllocationRequest, AssetAllocationResponse
import pandas as pd

def explain_recommendation(
    request: AssetAllocationRequest, response: AssetAllocationResponse
) -> str:
    explanations = []

    if request.risk_tolerance == "High":
        explanations.append(
            "You have a high risk tolerance, so more equity has been allocated to maximize growth."
        )
    elif request.risk_tolerance == "Low":
        explanations.append(
            "Since your risk tolerance is low, we've prioritized safer assets like debt and gold."
        )

    if request.goal_horizon_years < 5:
        explanations.append(
            "Your goal horizon is short-term, so less volatile assets are favored."
        )
    elif request.goal_horizon_years >= 10:
        explanations.append(
            "With a long-term goal, higher equity exposure is suitable for potential growth."
        )

    if request.investment_experience == "Beginner":
        explanations.append(
            "As you're a beginner, we've ensured a diversified and balanced portfolio."
        )

    if request.financial_goal == "Retirement":
        explanations.append(
            "Retirement goals require stable returns, so equity and debt are balanced accordingly."
        )

    if (
        request.income_per_month - request.monthly_expenses
    ) / request.income_per_month < 0.2:
        explanations.append(
            "Low savings ratio observed, so we've opted for a more conservative asset mix."
        )

    return " ".join(explanations)


def enforce_regulatory_safeguards(
    request: AssetAllocationRequest, response: AssetAllocationResponse
) -> (
    AssetAllocationResponse
):  # Adjust based on age and risk tolerance rules if request.age > 55 and response.equity_percent > 60: excess = response.equity_percent - 60 response.equity_percent = 60 response.debt_percent += excess

    if request.risk_tolerance == "Low" and response.equity_percent > 40:
        excess = response.equity_percent - 40
        response.equity_percent = 40
        response.debt_percent += excess

    # Normalize allocation to sum to 100
    total = (
        response.equity_percent
        + response.debt_percent
        + response.gold_percent
        + response.real_estate_percent
    )

    if total != 100:
        diff = 100 - total
        response.real_estate_percent += diff

    return response


def log_recommendation(
    request: AssetAllocationRequest, response: AssetAllocationResponse, explanation: str
):
    log_entry = {
        "age": request.age,
        "income": request.income_per_month,
        "expenses": request.monthly_expenses,
        "goal_horizon": request.goal_horizon_years,
        "risk": request.risk_tolerance,
        "goal": request.financial_goal,
        "experience": request.investment_experience,
        "equity": response.equity_percent,
        "debt": response.debt_percent,
        "gold": response.gold_percent,
        "real_estate": response.real_estate_percent,
        "explanation": explanation,
    }
    df = pd.DataFrame([log_entry])
    df.to_csv(
        "logs/recommendation_logs.csv",
        mode="a",
        header=not pd.io.common.file_exists("logs/recommendation_logs.csv"),
        index=False,
    )
