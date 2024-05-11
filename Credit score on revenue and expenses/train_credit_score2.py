import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your data
data = pd.read_csv('Credit-Scoring\Credit score on revenue and expenses\your_data.csv')
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

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
import joblib
joblib.dump(model, 'credit_score_model.pkl')