import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Load the data
file_path = 'P3- Churn-Modelling Data.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataframe
print(data.head())

# Customer Demographics
# Distribution of customers across different age groups
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribution of Customers by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Analyze the gender distribution of customers
plt.figure(figsize=(6, 6))
gender_distribution = data['Gender'].value_counts()
gender_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['skyblue', 'pink'])
plt.title('Gender Distribution of Customers')
plt.ylabel('')
plt.show()

# Churn Analysis
# Percentage of customers who have churned
churn_rate = data['churned'].value_counts(normalize=True) * 100
print(f"Churn Rate:\n{churn_rate}")

# Main reasons for customer churn (Correlation analysis)
correlation_matrix = data.corr()
print("Correlation Matrix with 'churned':\n", correlation_matrix['churned'].sort_values(ascending=False))

# Patterns among customers who have churned
churned_customers = data[data['churned'] == 1]
print(churned_customers.describe())

# Product Usage
# Most commonly used products or services (assuming 'NumOfProducts' column exists)
if 'NumOfProducts' in data.columns:
    plt.figure(figsize=(10, 6))
    product_distribution = data['NumOfProducts'].value_counts()
    sns.barplot(x=product_distribution.index, y=product_distribution.values)
    plt.title('Most Commonly Used Products or Services')
    plt.xlabel('NumOfProducts')
    plt.ylabel('Frequency')
    plt.show()

# Usage patterns of different customer segments
plt.figure(figsize=(10, 6))
sns.countplot(x='Geography', hue='churned', data=data)
plt.title('Churn by Geography')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()

# Financial Analysis
# Average account balance of customers
average_balance = data['Balance'].mean()
print(f"Average Account Balance: {average_balance}")

# # Financial characteristics of churned vs. non-churned customers
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='churned', y='Balance', data=data)
# plt.title('Account Balance by Churn Status')
# plt.xlabel('Churn Status')
# plt.ylabel('Account Balance')
# plt.show()z

# Encode categorical variables
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# Predictive Modeling
# Significant predictors of customer churn
features = data.drop(columns=['churned', 'CustomerId', 'Surname', 'RowNumber'])
target = data['churned']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
print("Feature Importances:\n", feature_importances)

# Predictive model performance
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Identify at-risk customers
at_risk_customers = data.iloc[X_test.index][y_pred == 1]
print("At-Risk Customers:\n", at_risk_customers.head())
