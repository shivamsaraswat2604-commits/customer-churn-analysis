import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(42)

# 1. GENERATE DATASET (400 Customer Records)
n_customers = 400

data = {
    'CustomerID': [f'CUST-{i:03d}' for i in range(1, n_customers + 1)],
    'Age': np.random.randint(18, 70, size=n_customers),
    'Tenure': np.random.randint(1, 72, size=n_customers),
    'ContractType': np.random.choice(
        ['Month-to-Month', 'One Year', 'Two Year'],
        size=n_customers,
        p=[0.5, 0.3, 0.2],
    ),
    'MonthlyCharges': np.round(
        np.random.uniform(20.0, 120.0, size=n_customers), 2
    ),
    'SupportCalls': np.random.randint(0, 6, size=n_customers),
    'PaymentMethod': np.random.choice(
        ['Credit Card', 'Bank Transfer', 'Electronic Check'], size=n_customers
    ),
}

df = pd.DataFrame(data)

# Calculate TotalCharges based on tenure and monthly charges with minor noise
df['TotalCharges'] = np.round(
    df['Tenure']
    * df['MonthlyCharges']
    * np.random.uniform(0.9, 1.1, size=n_customers),
    2,
)

# Simulate Churn based on logical business factors (e.g., month-to-month contracts & high charges increase churn)
churn_prob = (
    (df['ContractType'] == 'Month-to-Month') * 0.45
    + (df['MonthlyCharges'] > 85) * 0.20
    + (df['SupportCalls'] > 3) * 0.20
    - (df['Tenure'] > 24) * 0.30
)
churn_prob = np.clip(churn_prob, 0.05, 0.95)
df['Churn'] = np.random.binomial(1, churn_prob, size=n_customers)
df['Churn'] = df['Churn'].map({1: 'Yes', 0: 'No'})

# 2. DATA CLEANING & MISSING VALUE HANDLING
# Intentionally introduce missing values to demonstrate data cleaning skills
missing_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_indices[:12], 'TotalCharges'] = np.nan
df.loc[missing_indices[12:], 'MonthlyCharges'] = np.nan

print('--- Missing Values Before Cleaning ---')
print(df.isnull().sum())

# Handle missing values using median imputation
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['MonthlyCharges'] = df['MonthlyCharges'].fillna(
    df['MonthlyCharges'].median()
)

print('\n--- Missing Values After Cleaning ---')
print(df.isnull().sum())

# 3. EXPLORATORY DATA ANALYSIS (EDA)
print('\n--- Overall Churn Rate (%) ---')
print(df['Churn'].value_counts(normalize=True) * 100)

print('\n--- Churn Rate by Contract Type (%) ---')
contract_churn = (
    pd.crosstab(df['ContractType'], df['Churn'], normalize='index') * 100
)
print(contract_churn)

# 4. DATA VISUALIZATION
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Churn count by contract type
sns.countplot(
    data=df, x='ContractType', hue='Churn', palette='Set2', ax=axes[0]
)
axes[0].set_title('Customer Churn Count by Contract Type')
axes[0].set_xlabel('Contract Type')
axes[0].set_ylabel('Number of Customers')

# Plot 2: Monthly charges distribution by churn status
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='Set1', ax=axes[1])
axes[1].set_title('Monthly Charges Distribution vs Churn Status')
axes[1].set_xlabel('Churn')
axes[1].set_ylabel('Monthly Charges ($)')

plt.tight_layout()
plt.savefig('churn_analysis_plots.png')
print(
    '\n[INFO] Visualizations successfully generated and saved as "churn_analysis_plots.png".'
)
plt.show()

# 5. BUSINESS RECOMMENDATIONS
print('ACTIONABLE RETENTION RECOMMENDATIONS')
print(
    '1. Target Month-to-Month Customers: Month-to-month contracts experience the highest churn rate. Offer special discounted annual contracts to encourage migration.'
)
print(
    '2. Proactive Customer Support: Customers making more than 3 support calls have a high risk of leaving. Implement a VIP tech support or customer success outreach for these accounts.'
)
print(
    '3. Price Sensitivity Strategy: High monthly charges strongly correlate with churn. Provide tailored loyalty discounts or bundle services for high-bill customers.'
)