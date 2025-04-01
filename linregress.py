import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Data
cars_age = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
speed = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

# Reshaping data to 2D arrays
cars_age = cars_age.reshape(-1, 1)
speed = speed.reshape(-1, 1)

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(cars_age, speed, train_size=0.33, random_state=42)

# Creating the model and fitting it
lm = LinearRegression()
lm.fit(x_train, y_train)

# Predicting on test data
y_predict = lm.predict(x_test)

# Predicting for a new car age (10 years)
speed_of_car = np.array([[10]])
predicted_speed = lm.predict(speed_of_car)
print("The predicted speed for a car age 10 =", predicted_speed)

# Printing training data for verification
print("Training data (x_train):", x_train)
print("Training labels (y_train):", y_train)

# Plotting the results
plt.scatter(cars_age, speed, color='blue')  # Scatter plot of data
plt.plot(x_test, y_predict, color="black")  # Line plot for the regression line
plt.xlabel("Car Age")
plt.ylabel("Speed")
plt.show()

# Calculating the R2 score to measure the accuracy
print("R2 score =", r2_score(y_test, y_predict))
