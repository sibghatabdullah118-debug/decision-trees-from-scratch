import numpy as np
import pandas as pd


def generate(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    employment = rng.choice(["Salaried", "Self-Employed", "Unemployed"], n, p=[0.55, 0.30, 0.15])
    education = rng.choice(["High School", "Bachelor", "Master"], n, p=[0.3, 0.5, 0.2])
    city_tier = rng.choice(["Tier1", "Tier2", "Tier3"], n, p=[0.3, 0.4, 0.3])
    age = rng.integers(21, 66, n)
    income = np.round(rng.normal(55000, 20000, n).clip(15000, 150000), -2)
    credit_score = np.round(rng.normal(650, 80, n).clip(300, 850))

    df = pd.DataFrame({
        "employment": employment,
        "education": education,
        "city_tier": city_tier,
        "age": age,
        "income": income,
        "credit_score": credit_score,
    })

    # Hidden "true rule" that decides approval, plus some noise
    score = (
        0.02 * (df["credit_score"] - 650)
        + 0.00003 * (df["income"] - 55000)
        + df["employment"].map({"Salaried": 0.5, "Self-Employed": 0.0, "Unemployed": -1.5})
        + df["education"].map({"High School": -0.3, "Bachelor": 0.0, "Master": 0.3})
        + rng.normal(0, 0.5, n)
    )
    df["approved"] = np.where(score > 0, "Yes", "No")

    # Add a few missing values (C4.5 can handle these, ID3/CART need filling)
    df.loc[rng.choice(n, 30, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 30, replace=False), "employment"] = np.nan
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/loan_data.csv", index=False)
    print(df.head())
    print("\nApproval counts:\n", df["approved"].value_counts())
    print("\nMissing values:\n", df.isna().sum())