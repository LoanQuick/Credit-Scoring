# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import json

# app = Flask(__name__)

# @app.route('/predictlinear', methods=['POST'])
# def predictlinear():
#     # Load the input data from a JSON request
#     data_dict = request.get_json()
#     data = pd.DataFrame(data_dict)
#     data['date'] = pd.to_datetime(data['date'])

#     # Convert 'date' column to datetime and set as index
#     data.set_index('date', inplace=True)

#     # Aggregate to monthly data
#     monthly_data = data.resample('M').sum()
#     monthly_data['month_index'] = range(1, len(monthly_data) + 1)

#     # Train the models
#     model_revenue = LinearRegression().fit(monthly_data[['month_index']], monthly_data['daily_revenue'])
#     model_expenses = LinearRegression().fit(monthly_data[['month_index']], monthly_data['daily_expenses'])

#     # Predict for the next two months
#     last_month_index = monthly_data['month_index'].max()
#     predicted_revenues = model_revenue.predict([[last_month_index + 1], [last_month_index + 2]])
#     predicted_expenses = model_expenses.predict([[last_month_index + 1], [last_month_index + 2]])

#     # Create a dictionary with the predicted revenues and expenses
#     result = {
#         'predicted_revenues': predicted_revenues.tolist(),
#         'predicted_expenses': predicted_expenses.tolist()
#     }

#     # Return the result as JSON
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta
import json

app = Flask(__name__)

@app.route('/predict_month_revenue_expenses', methods=['POST'])
def predict_month_revenue_expenses():
    # Load the input data from a JSON request
    data_dict = request.get_json()
    data = pd.DataFrame(data_dict)
    data['date'] = pd.to_datetime(data['date'])

    # Convert 'date' column to datetime and set as index
    data.set_index('date', inplace=True)

    # Aggregate to monthly data
    monthly_data = data.resample('M').sum()

    # Create 'month_index' column
    monthly_data['month_index'] = range(1, len(monthly_data) + 1)

    # Train the models
    model_revenue = LinearRegression().fit(monthly_data[['month_index']], monthly_data['daily_revenue'])
    model_expenses = LinearRegression().fit(monthly_data[['month_index']], monthly_data['daily_expenses'])

    # Predict for the next two months
    last_month_index = monthly_data['month_index'].max()
    predicted_revenues = model_revenue.predict([[last_month_index + 1], [last_month_index + 2]])
    predicted_expenses = model_expenses.predict([[last_month_index + 1], [last_month_index + 2]])

    # Get the last date in the data
    last_date = monthly_data.index.max()

    # Create a dictionary with the predicted revenues and expenses
    result = {
        (last_date + relativedelta(months=i)).strftime('%Y-%m'): {
            'predicted_revenue': predicted_revenues[i-1],
            'predicted_expenses': predicted_expenses[i-1]
        } for i in range(1, 3)
    }

    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)