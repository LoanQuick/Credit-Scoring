import pandas as pd
from sklearn.linear_model import LinearRegression
import json

# Function to read data from JSON
def read_data_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to aggregate daily data to monthly
def aggregate_to_monthly(data):
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    monthly_data = df.resample('M').sum().reset_index()
    monthly_data['Month'] = monthly_data['date'].dt.to_period('M')
    return monthly_data

# Function to prepare data and train models
def train_models(data):
    df = data
    df['month_index'] = (df['date'].dt.year - 2022) * 12 + df['date'].dt.month - 1  # Adjust based on the starting year

    # Initialize models
    model_revenue = LinearRegression()
    model_expenses = LinearRegression()

    # Features and targets
    X = df[['month_index']]
    y_revenue = df['daily_revenue']
    y_expenses = df['daily_expenses']

    # Train the models
    model_revenue.fit(X, y_revenue)
    model_expenses.fit(X, y_expenses)
    
    return model_revenue, model_expenses

# Function to predict revenue and expenses for future months
def predict_future_months(model_revenue, model_expenses, last_index, months=2):
    future_indices = [[last_index + i] for i in range(1, months + 1)]
    predicted_revenues = model_revenue.predict(future_indices)
    predicted_expenses = model_expenses.predict(future_indices)
    return predicted_revenues, predicted_expenses

# Example usage
# Path to JSON file
json_file_path = 'csvjson (1).json'  # Change this to your actual file path

# Read and process data from JSON
data = read_data_from_json(json_file_path)
monthly_data = aggregate_to_monthly(data)

# Train the models
model_revenue, model_expenses = train_models(monthly_data)

# Predict for the next two months
last_month_index = monthly_data['month_index'].max()
predicted_revenues, predicted_expenses = predict_future_months(model_revenue, model_expenses, last_month_index, months=2)
print("Predicted Revenues:", predicted_revenues)
print("Predicted Expenses:", predicted_expenses)
