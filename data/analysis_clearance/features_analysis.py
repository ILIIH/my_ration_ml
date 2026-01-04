import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import normaltest

df_wide = pd.read_csv('cleaned_food_data.csv')

target_nutrients = ["Energy", "Protein", "Total lipid (fat)", "Carbohydrate, by difference"]

initial_count = len(df_wide)

for nutrient in target_nutrients:
    if nutrient in df_wide.columns:
        # Calculate IQR
        Q1 = df_wide[nutrient].quantile(0.25)
        Q3 = df_wide[nutrient].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers before removing
        outlier_mask = (df_wide[nutrient] < lower_bound) | (df_wide[nutrient] > upper_bound)
        num_outliers = outlier_mask.sum()
        
        # Filter the DataFrame to keep only non-outliers
        df_wide = df_wide[~outlier_mask]
        
        print(f"{nutrient}: Removed {num_outliers} outliers.")

final_count = len(df_wide)
print(f"\nTotal rows removed: {initial_count - final_count}")
print(f"Remaining rows in dataset: {final_count}")



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


for nutrient in target_nutrients:
    if nutrient in df_wide.columns:
        data = df_wide[nutrient].dropna()
        
        # Visualization
        plt.figure(figsize=(10, 4))
        
        # Histogram + KDE
        plt.subplot(1, 2, 1)
        sns.histplot(data, kde=True, color='skyblue')
        plt.title(f'Histogram: {nutrient}')
        
        # Q-Q Plot
        plt.subplot(1, 2, 2)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot: {nutrient}')
        
        plt.tight_layout()
        plt.show()

        #  The Stat Test (D'Agostino's K^2) ---
        stat, p = normaltest(data)
        
        print(f"--- Statistics for {nutrient} ---")
        print(f"Skewness: {data.skew():.2f}")
        print(f"Kurtosis: {data.kurtosis():.2f}")
        print(f"Normality Test p-value: {p:.4f}")
        
        if p < 0.05:
            print("Result: Data is NOT normally distributed (Reject H0)")
        else:
            print("Result: Data looks Normally distributed (Fail to reject H0)")
        print("-" * 30)