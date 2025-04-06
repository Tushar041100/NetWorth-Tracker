from pydantic import BaseModel, Field, field_validator
from typing import Literal
import numpy as np
import math

class AssetAllocationRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income_per_month: float = Field(..., gt=0)
    monthly_expenses: float = Field(..., ge=0)
    goal_horizon_years: int = Field(..., ge=0)
    risk_tolerance: Literal['Low', 'Medium', 'High']
    financial_goal: Literal['Retirement', 'Wealth Creation', 'Child Education', 'Travel']
    investment_experience: Literal['Beginner', 'Intermediate', 'Expert']
 
    @field_validator("monthly_expenses")
    @classmethod
    def validate_expenses(cls, v, info):
        income = info.data.get("income_per_month")
        if income is not None and v > income:
            raise ValueError("Expenses cannot exceed monthly income")
        return v
 
 
class AssetAllocationResponse(BaseModel):
    equity_percent: float = Field(..., ge=0, le=100)
    debt_percent: float = Field(..., ge=0, le=100)
    gold_percent: float = Field(..., ge=0, le=100)
    real_estate_percent: float = Field(..., ge=0, le=100)
 
    @field_validator("equity_percent", "debt_percent", "gold_percent", "real_estate_percent")
    @classmethod
    def validate_individual_percent(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Each allocation must be between 0 and 100")
        return v
 
    @field_validator("real_estate_percent")
    @classmethod
    def validate_total_percent(cls, v, info):
        data = info.data
        total = sum([
            data.get("equity_percent", 0) +
            data.get("debt_percent", 0) +
            data.get("gold_percent", 0) +
            v
        ])
        
        if not np.isclose(total, 100, atol=0.5):
            raise ValueError(f"Total allocation must sum up to 100%. Got: {total}%")
        return v