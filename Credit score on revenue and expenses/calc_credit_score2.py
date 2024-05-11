from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import json

app = Flask(__name__)

# Load the trained model
model = joblib.load('credit_score_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the input data from a JSON request
    data_dict = request.get_json()
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

    # Format the monthly credit scores
    monthly_credit_scores_dict = {str(date)[:7]: score for date, score in zip(X.index, monthly_credit_scores)}

    # Calculate the overall credit score
    overall_credit_score = monthly_credit_scores.mean()

    # Create a dictionary with the monthly and overall credit scores
    result = {
        'monthly_credit_scores': monthly_credit_scores_dict,
        'overall_credit_score': overall_credit_score
    }

    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)