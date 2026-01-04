import json
from constants import API_KEY, data_url
import requests
import re


# --- Fetching Logic (Same as before) ---
TARGET_TOTAL = 100000
PAGE_SIZE = 100 
all_foods = []
current_page = 1

while len(all_foods) < TARGET_TOTAL:
    payload = {
        "query": "food",
        "dataType": ["Survey (FNDDS)", "Foundation", "Branded"],
        "pageSize": PAGE_SIZE,
        "pageNumber": current_page,
        "nutrients": [
            {"nutrientId": 1008, "min": 100, "max": 200},
            {"nutrientId": 1003, "min": 5},
            {"nutrientId": 1004, "min": 0, "max": 10},
            {"nutrientId": 1005, "min": 10, "max": 40}
        ]
    }

    resp = requests.post(f"{data_url}?api_key={API_KEY}", json=payload)
    
    if resp.status_code == 200:
        page_results = resp.json().get('foods', [])
        if not page_results: break
        
        all_foods.extend(page_results)
        print(f"Collected: {len(all_foods)} items...")
        current_page += 1
    else:
        print(f"Error: {resp.status_code}")
        break

# Save to local 
FILENAME = "food_data.json"
final_data = all_foods[:TARGET_TOTAL]

with open(FILENAME, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

print(f"Successfully saved {len(final_data)} items to {FILENAME}")