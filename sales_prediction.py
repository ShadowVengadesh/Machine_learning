import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Read data from Excel file
df = pd.read_excel('sales_data.xlsx')
df.columns = ['sales', 'advertisement']
print(df.shape)
print(df.head())

# Defining features (x) and target (y)
x = df['sales'].values
y = df['advertisement'].values

# Plotting scatter plot
plt.scatter(x, y, color='blue', label='Scatter plot')
plt.xlabel('Sales')
plt.ylabel('Advertisement')
plt.title("Sales vs Advertisement")
plt.legend(loc=4)
plt.show()

# Reshaping x and y to fit the model
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
print(x.shape)

# Splitting the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Print shapes of train and test data
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Create a LinearRegression model and fit the training data
lm = LinearRegression()
lm.fit(x_train, y_train)

# Make predictions using the test data
y_pred = lm.predict(x_test)

# Output the slope (coefficient) and intercept of the model
a = lm.coef_
b = lm.intercept_
print("Slope =", a)
print("Intercept =", b)

# Predict the advertisement value for a sales value of 25
sales_value = np.array([[35]])  # Reshape the input to match the model's expected input shape
advertisement_value = lm.predict(sales_value)

print(f"For sales = 25, the predicted advertisement value is {advertisement_value[0][0]}")

print ("R2 Score value: {:.4f}".format(r2_score(y_test, y_pred)))

plt.scatter(x, y, color = 'blue', label='Scatter Plot')
plt.plot(x_test, y_pred, color = 'black', linewidth=3, label = 'Regression Line')
plt.title('Relationship between Sales and Advertising')
plt.xlabel('Sales')
plt.ylabel('Advertising')
plt.legend(loc=4)
plt.show()