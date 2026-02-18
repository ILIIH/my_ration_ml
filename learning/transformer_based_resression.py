import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel
import ast
from  data.pre_tokenized_food_dataset  import PreTokenizedFoodDataset
from torch.utils.data import Dataset, DataLoader

class FoodMacroRegressor(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(FoodMacroRegressor, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        self.kcal_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # Output: Kcal
        )
        
        self.protein_head = nn.Linear(hidden_size, 1)
        self.fat_head = nn.Linear(hidden_size, 1)
        self.carbs_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.pooler_output 
        
        kcal = self.kcal_head(pooled_output)
        protein = self.protein_head(pooled_output)
        fat = self.fat_head(pooled_output)
        carbs = self.carbs_head(pooled_output)
        
        return {
            'kcal': kcal,
            'protein': protein,
            'fat': fat,
            'carbs': carbs
        }

# Learning

data_set = pd.read_csv('../tokenization/ready_for_training.csv')

print("Converting string-lists back to integers...")
data_set['input_ids'] = data_set['input_ids'].apply(ast.literal_eval)
data_set = data_set.rename(columns={
    'Energy': 'kcal',
    'Protein': 'protein',
    'Total lipid (fat)': 'fat',
    'Carbohydrate, by difference': 'carbs'
})

targets = data_set[['kcal', 'protein', 'fat', 'carbs']].values

#  Create Dataset
dataset = PreTokenizedFoodDataset(data_set['input_ids'].tolist(), targets)

# Split into Train/Test (e.g., 80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#  Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

print(f"Ready! Training on {train_size} items, testing on {test_size} items.")