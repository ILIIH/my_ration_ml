import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Baseline model is median values or, previously selected values for some special categories

data_set = pd.read_csv('../data/analysis_clearance/cleaned_food_data.csv')

target_cols = [
    'Carbohydrate, by difference', 
    'Energy', 
    'Protein', 
    'Total lipid (fat)'
]

medians = data_set[target_cols].median()
baseline_values = medians.to_dict()

def baseline_model(description):
    desc = description.upper()
    
    if "CHICKEN" in desc or "TURKEY" in desc:
        return {"Carbs": 5.0, "Energy": 180, "Protein": 20.0, "Fat": 10.0}
    
    if "BREAD" in desc or "FLOUR" in desc:
        return {"Carbs": 50.0, "Energy": 250, "Protein": 8.0, "Fat": 3.0}
    
    if "BUTTER" in desc or "OIL" in desc:
        return {"Carbs": 0.0, "Energy": 700, "Protein": 0.0, "Fat": 80.0}

    # Fallback to dataset global median
    return {"Carbs": baseline_values['Carbohydrate, by difference'], "Energy": baseline_values['Energy'], "Protein": baseline_values['Protein'], "Fat": baseline_values['Total lipid (fat)'] }


tasks = {
    "Protein": ("Protein", "Protein"),
    "Carbohydrates": ("Carbohydrate, by difference", "Carbs"),
    "Energy": ("Energy", "Energy"),
    "Fat": ("Total lipid (fat)", "Fat")
}

for label, (csv_col, func_key) in tasks.items():
    y_true = data_set[csv_col].fillna(0).tolist() 
    y_pred = [baseline_model(desc)[func_key] for desc in data_set['description']]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- {label} ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")