import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the JSON file
file_path = 'P1- BankStatements.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract transactions
transactions = data['Account']['Transactions']['Transaction']

# Convert transactions to a DataFrame
df = pd.DataFrame(transactions)

# Convert relevant columns to appropriate data types
df['amount'] = df['amount'].astype(float)
df['currentBalance'] = df['currentBalance'].astype(float)
df['transactionTimestamp'] = pd.to_datetime(df['transactionTimestamp'])

# Transaction Analysis
total_transactions = len(df)
print("Total number of transactions:", total_transactions)

# Define small and large transactions
df['transaction_size'] = df['amount'].apply(lambda x: 'small' if x < 500 else 'large')
transaction_distribution = df['transaction_size'].value_counts()
print("Transaction distribution:\n", transaction_distribution)

transaction_types = df['type'].value_counts()
print("Frequency of transaction types:\n", transaction_types)

# Balance Analysis
df.set_index('transactionTimestamp', inplace=True)
df['currentBalance'].plot(title="Account Balance Over Time")
plt.xlabel('Date')
plt.ylabel('Balance (INR)')
plt.show()

# Significant changes in balance (define significant change as > 1000)
df['balance_change'] = df['currentBalance'].diff().abs()
significant_changes = df[df['balance_change'] > 1000]
print("Periods with significant changes in account balance:\n", significant_changes)

# Spending Patterns
def categorize_expense(narration):
    if 'FILLING' in narration or 'PETROL' in narration:
        return 'Fuel'
    elif 'SHOP' in narration or 'MART' in narration:
        return 'Shopping'
    elif 'ATM' in narration:
        return 'ATM Withdrawal'
    elif 'UPI' in narration:
        return 'UPI'
    elif 'FOOD' in narration or 'RESTAURANT' in narration:
        return 'Food'
    else:
        return 'Other'

df['expense_category'] = df['narration'].apply(categorize_expense)
expense_categories = df[df['type'] == 'DEBIT']['expense_category'].value_counts()
print("Expense categories:\n", expense_categories)

category_spending = df[df['type'] == 'DEBIT'].groupby('expense_category')['amount'].agg(['count', 'sum'])
print("Spending in each category:\n", category_spending)

# Income Analysis
def categorize_income(narration):
    if 'SALARY' in narration:
        return 'Salary'
    elif 'UPI' in narration:
        return 'UPI'
    else:
        return 'Other'

df['income_category'] = df['narration'].apply(categorize_income)
income_sources = df[df['type'] == 'CREDIT']['income_category'].value_counts()
print("Income sources:\n", income_sources)

# Group by month to find patterns in timing and amount of income
income_timing = df[df['type'] == 'CREDIT'].resample('M')['amount'].sum()
print("Income timing:\n", income_timing)
income_timing.plot(title="Monthly Income Received")
plt.xlabel('Month')
plt.ylabel('Income (INR)')
plt.show()

# Alert Generation
# Unusual or suspicious transactions (amount > 5000)
suspicious_transactions = df[df['amount'] > 5000]
print("Suspicious transactions:\n", suspicious_transactions)

# Alerts for low balance or high expenditure periods
low_balance_alerts = df[df['currentBalance'] < 500]
print("Low balance alerts:\n", low_balance_alerts)

daily_expenditure = df[df['type'] == 'DEBIT'].resample('D')['amount'].sum()
high_expenditure_alerts = daily_expenditure[daily_expenditure > 2000]
print("High expenditure alerts:\n", high_expenditure_alerts)
