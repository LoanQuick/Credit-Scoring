import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import json

# Load the trained model
model = joblib.load('Credit-Scoring\Credit score on revenue and expenses\credit_score_model.pkl')

# Load the input data
# data = pd.read_csv('Credit-Scoring\Credit score on revenue and expenses\your_data.csv')
with open('Credit-Scoring\Credit score on revenue and expenses\csvjson (1).json', 'r') as f:
    data_dict = json.load(f)
data = pd.DataFrame(data_dict)
data['date'] = pd.to_datetime(data['date'])

# Set 'date' as index and ensure it's datetime
data.set_index('date', inplace=True)

# Feature Engineering
data['daily_net_income'] = data['daily_revenue'] - data['daily_expenses']
data['revenue_to_expense_ratio'] = data['daily_revenue'] / data['daily_expenses']

# Aggregate to monthly data
monthly_data = data.resample('M').sum()
monthly_data['average_daily_revenue'] = data['daily_revenue'].resample('M').mean()
monthly_data['average_daily_expenses'] = data['daily_expenses'].resample('M').mean()

# Prepare the dataset for prediction
X = monthly_data[['daily_net_income', 'revenue_to_expense_ratio', 'average_daily_revenue', 'average_daily_expenses']]

# Predict the credit scores
monthly_credit_scores = model.predict(X)

# Print the monthly credit scores
print("Monthly Credit Scores:")
for date, score in zip(monthly_data.index, monthly_credit_scores):
    print(f"{date.strftime('%Y-%m')}: {score:.2f}")

# Calculate and print the overall credit score
overall_credit_score = monthly_credit_scores.mean()
print(f"\nOverall Credit Score: {overall_credit_score:.2f}")
