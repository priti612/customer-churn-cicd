import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean invalid numeric values and add customer-behaviour features."""
    numeric_limits = {
        "Tenure_Months": (0, 72),
        "Monthly_Charges_INR": (0, None),
        "Total_Charges_INR": (0, None),
        "Support_Tickets_6M": (0, None),
        "Avg_Data_Usage_GB": (0, None),
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (low, high) in self.numeric_limits.items():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X.loc[X[col] < low, col] = np.nan
                if high is not None:
                    X.loc[X[col] > high, col] = np.nan

        tenure = X["Tenure_Months"]
        monthly = X["Monthly_Charges_INR"]
        total = X["Total_Charges_INR"]
        tickets = X["Support_Tickets_6M"]
        usage = X["Avg_Data_Usage_GB"]

        X["Tenure_Category"] = pd.cut(
            tenure, bins=[-np.inf, 12, 36, np.inf],
            labels=["New", "Regular", "Loyal"], right=True
        ).astype("object")
        X["Average_Monthly_Spending"] = total / tenure.replace(0, np.nan)
        X["Monthly_Total_Charge_Ratio"] = monthly / total.replace(0, np.nan)
        X["Ticket_Intensity"] = tickets / 6.0
        X["Service_Usage_Indicator"] = np.where(usage > 0, "Active", "Low_or_None")
        return X
