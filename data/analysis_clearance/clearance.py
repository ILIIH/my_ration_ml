import json
import pandas as pd

TARGET_NUTRIENTS = [
    "Energy", 
    "Protein", 
    "Total lipid (fat)", 
    "Carbohydrate, by difference"
]

with open('../source/food_data.json', 'r') as f:
    data = json.load(f)

processed_data = []

# Filter items
for item in data:
    filtered_nutrients = [
        {
            "nutrientName": n.get("nutrientName"),
            "value": n.get("value"),
            "unitName": n.get("unitName"),
        }
        for n in item.get("foodNutrients", [])
        if n.get("nutrientName") in TARGET_NUTRIENTS
    ]

    if filtered_nutrients:
        processed_data.append({
            "description": item.get("description"),
            "foodNutrients": filtered_nutrients
        })

df_flat = pd.json_normalize(
    processed_data, 
    record_path=['foodNutrients'], 
    meta=['description']
)

# Save to a flattened CSV for a clean table view

df_flat = df_flat[['description', 'nutrientName', 'value', 'unitName']]
df_flat.to_csv('filtered_food_data.csv', index=False)

