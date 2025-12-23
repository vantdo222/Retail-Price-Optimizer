# scheduler.py

import pandas as pd
import sys
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Error: gurobipy is not installed. Please install it to run the optimization.")
    sys.exit(1)

# ---------- Helper Function from Notebook ----------
def parse_ladders(rules_df, products):
    """
    Parse a simple ladder rule string like:
    'PremiumDetergent >= ValueDetergent + 3.00'
    into a list of tuples: [(premium, base, gap)].
    """
    ladders = []
    # Check if column exists to avoid errors if sheet format differs slightly
    if 'BusinessRule' not in rules_df.columns:
        return ladders

    row = rules_df[rules_df['BusinessRule'] == 'PriceLaddering']
    if row.empty:
        return ladders
    
    val = str(row['Value'].iloc[0]).strip()
    if val.lower() == 'none':
        return ladders
    
    # Expected format: Premium >= Base + gap
    try:
        left, right = val.split(">=")
        premium = left.strip()
        base_part, gap_part = right.split("+")
        base = base_part.strip()
        gap = float(gap_part.strip())
        if premium in products and base in products:
            ladders.append((premium, base, gap))
    except Exception:
        pass
    
    return ladders

# ---------- Main Optimize Function ----------
def optimize(input_file, output_file, k=3, mode="optimal", time_limit=300.0, verbose=False):
    """
    Runs the Gurobi optimization logic extracted from the notebook.
    """
    if verbose:
        print(f"Reading input file: {input_file}...")

    # -------- 1. READ DATA --------
    try:
        product_df = pd.read_excel(input_file, sheet_name="Product Info")
        demand_df = pd.read_excel(input_file, sheet_name="Demand Parameters")
        gamma_df  = pd.read_excel(input_file, sheet_name="Cross-Price Gamma")
        comp_df   = pd.read_excel(input_file, sheet_name="Competitor Map")
        rules_df  = pd.read_excel(input_file, sheet_name="Business Rules")
    except ValueError as e:
        raise ValueError(f"Input file missing required sheets (Product Info, Demand Parameters, etc.). Error: {e}")

    # Set of products
    I = list(product_df['Product'])

    # Basic dictionaries
    P = product_df.set_index('Product')['CurrentPrice'].to_dict()
    C = product_df.set_index('Product')['Cost'].to_dict()
    B = product_df.set_index('Product')['BrandMinPrice'].to_dict()
    Q = product_df.set_index('Product')['Inventory'].to_dict()
    
    alpha = demand_df.set_index('Product')['Alpha'].to_dict()
    beta  = demand_df.set_index('Product')['Beta'].to_dict()
    
    # Business rule: price stability limit
    row = rules_df[rules_df['BusinessRule'] == 'PriceStabilityLimit']
    if row.empty:
        max_change = 0.20
    else:
        max_change = float(row['Value'].iloc[0])
    
    # Business rule: price ladders
    ladders = parse_ladders(rules_df, I)
    
    # Competitor map J[i]: list of competitors for product i
    J = {}
    for _, r in comp_df.iterrows():
        prod = r['Product']
        comp_str = str(r.iloc[1]).strip()
        if comp_str.lower() == "none":
            J[prod] = []
        else:
            comps = [c.strip() for c in comp_str.split(",")]
            J[prod] = [c for c in comps if c in I]
    
    # Gamma: build gamma[j] for each competitor j
    gamma = {}
    for col in gamma_df.columns[1:]:
        vals = pd.to_numeric(gamma_df[col], errors='coerce').dropna()
        if len(vals) > 0:
            gamma[col] = float(vals.mean())
        else:
            gamma[col] = 0.0
    
    # -------- 2. MODEL SETUP --------
    mod = gp.Model()
    mod.setParam("NonConvex", 2)
    mod.setParam("TimeLimit", time_limit) # Use the time limit passed from GUI
    
    if not verbose:
        mod.setParam("OutputFlag", 0)
    else:
        mod.setParam("OutputFlag", 1)
    
    # Decision variables
    x = mod.addVars(I, name='Price')
    y = mod.addVars(I, vtype=GRB.INTEGER, name='PsychBase')
    z = mod.addVars(I, vtype=GRB.BINARY, name='UsePsychPrice')

    # Objective: Maximize total profit
    mod.setObjective(
        sum(
            (x[i] - C[i]) * (
                alpha[i]
                - beta[i] * x[i]
                + sum(gamma[j] * x[j] for j in J.get(i, []))
            )
            for i in I
        ),
        GRB.MAXIMIZE
    )
    
    # -------- 3. CONSTRAINTS --------
    
    for i in I:
        # Psych pricing vs Current Price logic
        mod.addConstr((z[i] == 1) >> (x[i] == y[i] + 0.99), name=f"Psych_Active_{i}")
        mod.addConstr((z[i] == 0) >> (x[i] == P[i]), name=f"Keep_Current_{i}")

    # Max price <= Current Price
    mod.addConstrs((x[i] <= P[i] for i in I), name='MaxPrice_Original')

    # Min price set by brand
    mod.addConstrs((x[i] >= B[i] for i in I), name='MinPrice_Brand')
    
    # Inventory: demand <= inventory
    mod.addConstrs(
        (
            Q[i] >= alpha[i]
                   - beta[i] * x[i]
                   + sum(gamma[j] * x[j] for j in J.get(i, []))
            for i in I
        ),
        name='Inventory'
    )
    
    # Price stability
    mod.addConstrs((x[i] >= P[i] * (1 - max_change) for i in I), name='Stability_Lower')
    
    # Price laddering
    for premium, base, gap in ladders:
        if premium in I and base in I:
            mod.addConstr(x[premium] >= x[base] + gap, name=f"Ladder_{premium}_{base}")
    
    # -------- 4. SOLVE --------
    if verbose:
        print("Solving Gurobi model...")
    mod.optimize()
    
    # -------- 5. OUTPUT --------
    results_list = []
    
    if mod.status == GRB.OPTIMAL or mod.status == GRB.TIME_LIMIT:
        print(f"Solution found. Objective Value: {mod.ObjVal:.2f}")
        
        current_total_profit = 0
        new_total_profit = mod.ObjVal

        for i in I:
            old_p = P[i]
            new_p = x[i].X
            is_psych = z[i].X > 0.5
            strat = "Psych (.99)" if is_psych else "Keep Current"
            pct_change = (new_p - old_p) / old_p * 100
            
            # Calculate final demand and profit per product for report
            demand = alpha[i] - beta[i] * new_p + sum(gamma[j] * x[j].X for j in J.get(i, []))
            profit = (new_p - C[i]) * demand
            
            # Calculate what current profit would be (for comparison)
            cur_demand = alpha[i] - beta[i] * P[i] + sum(gamma[j] * P[j] for j in J.get(i, []))
            cur_profit = (P[i] - C[i]) * cur_demand
            current_total_profit += cur_profit

            results_list.append({
                "Product": i,
                "Old Price": round(old_p, 2),
                "New Price": round(new_p, 2),
                "Strategy": strat,
                "Change %": round(pct_change, 2),
                "Expected Demand": round(demand, 2),
                "Expected Profit": round(profit, 2)
            })
            
        # Create DataFrame
        df_out = pd.DataFrame(results_list)
        
        # Add summary row or print to logs
        print("\n--- Profit Analysis ---")
        print(f"Current Profit (est): ${current_total_profit:,.2f}")
        print(f"Optimal Profit:       ${new_total_profit:,.2f}")
        print(f"Profit Lift:          ${new_total_profit - current_total_profit:,.2f}")

        # Save to Excel
        df_out.to_excel(output_file, index=False)
        print(f"Results written to {output_file}")
        
        return {"rows": len(df_out), "output_file": output_file}

    else:
        raise Exception("Optimization failed or was infeasible.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--mode", default="optimal")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    optimize(args.input_file, args.output_file, k=args.k, mode=args.mode, time_limit=args.time_limit, verbose=args.verbose)