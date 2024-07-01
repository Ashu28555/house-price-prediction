# House-price-prediction
# Task 1 : Implement a linear regression model to predict the prices of houses based on their sqaure footage and the number of bedrooms and bathrooms .

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv("train.csv")

# Display the dataset
data

# Display the shape of the dataset
data.shape

# Display dataset information
data.info()

# Display descriptive statistics
data.describe()

# Visualize relationships
sns.scatterplot(x='GrLivArea', y='SalePrice', data=data)
plt.title('Sale Price vs. GrLivArea')

sns.scatterplot(x='BedroomAbvGr', y='SalePrice', data=data)
plt.title('Sale Price vs. BedroomAbvGr')

sns.scatterplot(x='FullBath', y='SalePrice', data=data)
plt.title('Sale Price vs. FullBath')

sns.scatterplot(x='HalfBath', y='SalePrice', data=data)
plt.title('Sale Price vs. HalfBath')

# Select features and target variable
x = data[['GrLivArea', 'BedroomAbvGr', 'FullBath','HalfBath']]
y = data['SalePrice']

x

y

# Check for missing values
x.isna().sum()

y.isna().sum()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train

x_test

# Train Linear Regression model
model = LinearRegression()

model.fit(x_train,y_train)

# Predict with Linear Regression model
pred = model.predict(x_test)

pred

# Evaluate Linear Regression model
mse = mean_squared_error(y_test,pred)
print("The mean Squared error is :",mse)

r2 = r2_score(y_test,pred)
print("The R2 score is : ",r2)

# Plot actual vs predicted prices for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test,pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Best fit line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Train RandomForestRegressor model
model2 = RandomForestRegressor()

model2.fit(x_train,y_train)

# Predict with RandomForestRegressor model
pred2 = model2.predict(x_test)

pred2

# Evaluate RandomForestRegressor model
mse = mean_squared_error(y_test,pred2)
print("The mean Squared error is :",mse)

r2 = r2_score(y_test,pred2)
print("The R2 score is : ",r2)

# Cross-validation for Random forest Regression model
cv_scores = cross_val_score(model2, x, y, cv=5)
print("Cross-validation R2 scores:", cv_scores)
print("Mean cross-validation R2 score:", np.mean(cv_scores))

# Plot actual vs predicted prices for RandomForestRegressor
plt.figure(figsize=(10, 6))
plt.scatter(y_test, pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Best fit line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# The End

