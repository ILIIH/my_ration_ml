import pandas as pd
from transformers import BertTokenizer

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Priority Logic (The Weighting)
PRIORITY_MAP = {
    "PRIMARY": ["BREAD", "BUN", "LOAF", "BAGUETTE", "ROLL", "SLICE"],
    "SECONDARY": ["MULTI SEED", "WHOLEGRAIN", "RYE", "SOURDOUGH", "WHITE", "WHEAT"],
    "TERTIARY": ["ORGANIC", "SMALL BATCH", "HANDMADE", "FRESH", "TOASTED"]
}

def create_structured_input(text_string):
    """
    Input: Single string (e.g., "Organic Multi Seed Bread")
    Output: Structured string (e.g., "Product: BREAD | Type: MULTI SEED...")
    """
    if not isinstance(text_string, str):
        return "" 
        
    text_upper = text_string.upper()
    
    # Identify keywords present in this specific string
    found_primary = [w for w in PRIORITY_MAP["PRIMARY"] if w in text_upper]
    found_secondary = [w for w in PRIORITY_MAP["SECONDARY"] if w in text_upper]
    
    parts = []
    
    # Priority 1: Main Product
    if found_primary:
        parts.append(f"Product: {' '.join(found_primary)}")
        
    # Priority 2: Type/Feature
    if found_secondary:
        parts.append(f"Type: {' '.join(found_secondary)}")

    parts.append(f"Context: {text_string}")
    return " | ".join(parts)


# 1. Load  CSV file
df = pd.read_csv('../data/analysis_clearance/cleaned_food_data.csv')
print("CSV loaded successfully.")

# Apply Weighting Structure (String -> String)
print("Applying weighting structure...")
df["structured_text"] = df["description"].apply(create_structured_input)

# Tokenize (String -> List of Integers)
print("Tokenizing data...")
df["input_ids"] = df["structured_text"].apply(
    lambda x: tokenizer.encode(
        x, 
        add_special_tokens=True, # Adds [CLS] and [SEP]
        truncation=True, 
        max_length=64,           # Limit length to keep data uniform
        padding='max_length'     # Pad shorter strings with 0s so all lists are same length
    )
)

# Save to CSV
output_file = "ready_for_training.csv"
df.to_csv(output_file, index=False)