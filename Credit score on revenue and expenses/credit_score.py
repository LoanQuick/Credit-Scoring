import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your data
data = pd.read_csv('Credit-Scoring/Credit score on revenue and expenses/Sample_Credit_Score_Data_Six_Months.csv')
data['date'] = pd.to_datetime(data['date'])

# Feature Engineering
data['daily_net_income'] = data['daily_revenue'] - data['daily_expenses']
data['revenue_to_expense_ratio'] = data['daily_revenue'] / data['daily_expenses']

# Aggregate to monthly data by setting date as index
data.set_index('date', inplace=True)
monthly_data = data.resample('M').sum()
monthly_data['average_daily_revenue'] = data.resample('M')['daily_revenue'].mean()
monthly_data['average_daily_expenses'] = data.resample('M')['daily_expenses'].mean()
monthly_data['credit_score'] = data['credit_score'].resample('M').mean()

# Check if there's enough data to split
if len(monthly_data) > 1:
    X = monthly_data[['daily_net_income', 'revenue_to_expense_ratio', 'average_daily_revenue', 'average_daily_expenses']]
    y = monthly_data['credit_score']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
else:
    print("Not enough data to perform train-test split.")

# Example prediction (modify as needed if no split)
if 'model' in locals():
    example_data = pd.DataFrame({
        'daily_net_income': [20000],
        'revenue_to_expense_ratio': [1.5],
        'average_daily_revenue': [10000],
        'average_daily_expenses': [8000]
    })
    credit_score_prediction = model.predict(example_data)
    print(f"Predicted Credit Score: {credit_score_prediction[0]}")
else:
    print("Insufficient data for prediction.")
