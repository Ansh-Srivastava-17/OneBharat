import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'P2- OfficeSupplies Data.csv'
df = pd.read_csv(file_path)

# Convert OrderDate to datetime
df['OrderDate'] = pd.to_datetime(df['OrderDate'], format='%d-%b-%y')

# Calculate total sales for each row
df['Total Sales'] = df['Units'] * df['Unit Price']

# 1. Sales Analysis
# Total sales for each product category
total_sales_by_category = df.groupby('Item')['Total Sales'].sum().sort_values(ascending=False)
print("Total sales for each product category:\n", total_sales_by_category)

# Product category with the highest sales
highest_sales_category = total_sales_by_category.idxmax()
print("Product category with the highest sales:", highest_sales_category)

# Top 10 best-selling products (by total units sold)
top_10_best_selling_products = df.groupby('Item')['Units'].sum().sort_values(ascending=False).head(10)
print("Top 10 best-selling products:\n", top_10_best_selling_products)

# 2. Customer Analysis
# Top 10 customers by sales
top_10_customers = df.groupby('Rep')['Total Sales'].sum().sort_values(ascending=False).head(10)
print("Top 10 customers by sales:\n", top_10_customers)

# Total number of unique customers
total_unique_customers = df['Rep'].nunique()
print("Total number of unique customers:", total_unique_customers)

# Customer purchase frequency
customer_purchase_frequency = df['Rep'].value_counts()
print("Customer purchase frequency:\n", customer_purchase_frequency)

# 3. Time Series Analysis
# Monthly sales trends over the past year
df.set_index('OrderDate', inplace=True)
monthly_sales_trends = df['Total Sales'].resample('M').sum()
print("Monthly sales trends:\n", monthly_sales_trends)
monthly_sales_trends.plot(title="Monthly Sales Trends")
plt.xlabel('Month')
plt.ylabel('Total Sales (INR)')
plt.show()

# Identify any seasonal patterns in the sales data
monthly_sales_trends.groupby(monthly_sales_trends.index.month).mean().plot(title="Average Monthly Sales")
plt.xlabel('Month')
plt.ylabel('Average Sales (INR)')
plt.show()

# 4. Geographical Analysis
# Regions generating the most sales
sales_by_region = df.groupby('Region')['Total Sales'].sum().sort_values(ascending=False)
print("Regions generating the most sales:\n", sales_by_region)

# Sales trends across different regions
sales_trends_by_region = df.groupby(['Region', df.index.to_period('M')])['Total Sales'].sum().unstack(level=0)
print("Sales trends across different regions:\n", sales_trends_by_region)
sales_trends_by_region.plot(title="Sales Trends by Region")
plt.xlabel('Month')
plt.ylabel('Total Sales (INR)')
plt.legend(title='Region')
plt.show()

# 5. Profit Analysis
# For this analysis, assume a fixed profit margin of 20% on the unit price
df['Profit'] = df['Total Sales'] * 0.20

# Total profit for each product category
total_profit_by_category = df.groupby('Item')['Profit'].sum().sort_values(ascending=False)
print("Total profit for each product category:\n", total_profit_by_category)

# Top 10 most profitable products
top_10_profitable_products = df.groupby('Item')['Profit'].sum().sort_values(ascending=False).head(10)
print("Top 10 most profitable products:\n", top_10_profitable_products)
