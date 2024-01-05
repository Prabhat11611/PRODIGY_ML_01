import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('house_data.csv')

# Split the dataset into features and target variable
X = data[['square_footage', 'bedrooms', 'bathrooms']]
y = data['price']

# Create an instance of the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X, y)

# Predict the prices of houses
new_data = pd.DataFrame({'square_footage': [1500], 'bedrooms': [3], 'bathrooms': [2]})
predicted_prices = model.predict(new_data)

print(predicted_prices)