from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json
import joblib

app = Flask(__name__)

@app.route('/train_model_predict_score', methods=['POST'])
def train_model_predict_score():
    # Load the input data from a JSON request
    data_dict = request.get_json()
    data = pd.DataFrame(data_dict)
    data['date'] = pd.to_datetime(data['date'])

    # Convert 'date' column to datetime and set as index
    data.set_index('date', inplace=True)

    # Feature Engineering
    data['daily_net_income'] = data['daily_revenue'] - data['daily_expenses']
    data['revenue_to_expense_ratio'] = data['daily_revenue'] / data['daily_expenses']

    # Aggregate to monthly data
    monthly_data = data.resample('M').sum()
    monthly_data['average_daily_revenue'] = data['daily_revenue'].resample('M').mean()
    monthly_data['average_daily_expenses'] = data['daily_expenses'].resample('M').mean()

    # Load credit scores, ensuring it's only numeric values
    credit_scores = pd.read_csv('Credit-Scoring\Credit score on revenue and expenses\your_credit_scores.csv')
    credit_scores['date'] = pd.to_datetime(credit_scores['date'])
    credit_scores.set_index('date', inplace=True)

    # Prepare the dataset
    X = monthly_data[['daily_net_income', 'revenue_to_expense_ratio', 'average_daily_revenue', 'average_daily_expenses']]
    y = credit_scores['credit_score']  # Ensuring y is purely numeric

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'credit_score_model.pkl')

    # Return a success message
    return jsonify({'message': 'Model trained successfully'})

if __name__ == '__main__':
    app.run(debug=True)