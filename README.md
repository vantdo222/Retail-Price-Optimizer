# 🏷️ Retail Price Optimization Engine (MILP)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Solver](https://img.shields.io/badge/Solver-Gurobi-red)

## 📖 Overview
This project is an algorithmic pricing engine designed to maximize retail profit margins for a seasonal product catalog. 

Unlike simple "cost-plus" pricing strategies, this engine uses **Mixed-Integer Linear Programming (MILP)** to determine the mathematically optimal price point for every product. It balances **Price Elasticity of Demand**, competitor moves, and psychological pricing tactics ("sticky prices") to find the global optimum.

The core logic is built in Python using the **Gurobi Optimizer**.

## 🚀 Key Features

### 1. Demand Modeling (Price Elasticity)
The model forecasts unit sales ($D$) based on a linear demand curve calibrated on historical data:
$$D = \alpha - \beta \cdot P + \sum (\gamma \cdot P_{comp})$$
* **$\alpha$ (Alpha):** Base demand intercept.
* **$\beta$ (Beta):** Own-price sensitivity (elasticity).
* **$\gamma$ (Gamma):** Cross-price elasticity (impact of competitor pricing).

### 2. Psychological Pricing (Integer Constraints)
The solver includes binary variables to enforce "charm pricing" (e.g., ending in .99) where applicable. It weighs the trade-off between the margin lost by dropping to .99 vs. the volume gained by the psychological appeal.

### 3. Business Guardrails
To ensure practical viability, the model enforces:
* **Inventory Constraints:** $Demand \leq Current Inventory$.
* **Price Laddering:** Ensures premium products remain more expensive than base products (e.g., *Brand_Gold > Brand_Silver + $2.00*).
* **Stability Limits:** Prevents drastic price cuts (e.g., max drop of 20%) to protect brand value.

## 🛠️ Technology Stack
* **Language:** Python
* **Solver:** Gurobi (gurobipy)
* **Data Processing:** Pandas, NumPy, OpenPyXL
* **Input/Output:** Excel (.xlsx) integration for business-user friendliness.

## 📂 Repository Structure
```text
├── data/
│   └── Walmart_Seasonal_Large.xlsx   # Input data (Costs, Demand parameters, Competitors)
├── src/
│   └── scheduler.py                  # Main optimization logic
├── output/
│   └── optimized_prices.xlsx         # Generated results (New prices, Lift analysis)
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation
