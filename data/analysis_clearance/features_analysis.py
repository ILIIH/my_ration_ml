import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_wide = pd.read_csv('cleaned_food_data.csv')

# OUTLIERS (IQR Method)
print("\n--- Outlier Detection ---")
target_nutrients = ["Energy", "Protein", "Total lipid (fat)", "Carbohydrate, by difference"]

for nutrient in target_nutrients:
    if nutrient in df_wide.columns:
        # Calculate IQR
        Q1 = df_wide[nutrient].quantile(0.25)
        Q3 = df_wide[nutrient].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = df_wide[(df_wide[nutrient] < lower_bound) | (df_wide[nutrient] > upper_bound)]
        
        print(f"{nutrient}: Found {len(outliers)} potential outliers.")
        if not outliers.empty:
            print(f"   Bounds: {lower_bound:.2f} to {upper_bound:.2f}")



# DUPLICATE DETECTION 
subset_duplicates = df_wide.duplicated(subset=['description']).sum()
print(f"Duplicates based on product description: {subset_duplicates}")

# BOXPLOTS 

plt.figure(figsize=(6, 8))
sns.boxplot(y=df_wide['Energy'])
plt.title('Distribution of Energy')
plt.show()


plt.figure(figsize=(6, 8))
sns.boxplot(y=df_wide['Protein'])
plt.title('Distribution of Protein')
plt.show()

plt.figure(figsize=(6, 8))
sns.boxplot(y=df_wide['Total lipid (fat)'])
plt.title('Distribution of lipid')
plt.show()

plt.figure(figsize=(6, 8))
sns.boxplot(y=df_wide['Carbohydrate, by difference'])
plt.title('Distribution of carbohydrate')
plt.show()