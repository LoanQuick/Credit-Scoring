from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import json
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data1 = pd.read_csv("train_predict_credit.csv")
data2= pd.read_csv("train_predict_score.csv")
# # Placeholder to store the trained model1
model1 = joblib.load('credit_predict_model.pkl')
model2 = joblib.load('credit_score_model.pkl')
# For saving and loading sklearn objects
app = Flask(__name__)

# Global dictionary to store encoders for each feature
encoders = {}

@app.route('/train_model_predict_credit', methods=['POST'])
def train_model_predict_credit():
    global model1, data1, encoders
    # Target variable is hardcoded for simplicity
    target_variable = "Receive/ Not receive credit"
    X = data1.drop(target_variable, axis=1)
    y = data1[target_variable]
    
    # Encoding categorical features and saving encoders
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            encoders[column] = le  # Store the encoder
    
    # Save the encoders for future use
    joblib.dump(encoders, 'credit_predict_model.pkl')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    # Train the model1
    model1 = LogisticRegression(random_state=0, class_weight='balanced')
    # model1.fit(X, y)
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # return jsonify({"message": "Model trained"}), 200
    return jsonify({"message": "Model trained", "accuracy": accuracy}), 200

@app.route('/predict_credit', methods=['POST'])
def predict_credit():
    global model1, encoders
    
    # Load encoders
    encoders = joblib.load('credit_predict_model.pkl')

    # Process input data
    input_data = request.get_json()
    test_data = pd.DataFrame([input_data])
    
    # Apply the same encoders used during training
    for column in test_data.columns:
        if test_data[column].dtype == 'object':
            le = encoders[column]
            test_data[column] = le.transform(test_data[column])

    # Prediction using the model1
    prediction = model1.predict(test_data)
    prediction_result = True if prediction[0] == 1 else False
    return jsonify({"prediction": prediction_result})

@app.route('/train_model_predict_score', methods=['POST'])
def train_model_predict_score():
    # Load the input data from a JSON request
    data = pd.DataFrame(data2)
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
    credit_scores = pd.read_csv('train_my_predict_credit_scores.csv')
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
    return jsonify({'message': 'Model trained successfully'}), 200

@app.route('/predict_score', methods=['POST'])
def predict_score():
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
    monthly_credit_scores = model2.predict(X)

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
