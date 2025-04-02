import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Load dataset
df = pd.read_csv("co2.csv")
print(df.head())

# Scatter plot of CO2 emissions
df.plot.scatter(x="year", y="co2")
plt.xlabel("Years")
plt.ylabel("CO2 Emissions")
plt.title("Annual Carbon Emissions of China")
plt.show()

# Preparing data
x = df['year'].values.reshape(-1, 1)
y = df['co2'].values.reshape(-1, 1)

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Transforming features into polynomial form (degree=2)
poly = PolynomialFeatures(degree=3)
x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)

# Creating and training the polynomial regression model
lm = LinearRegression()
lm.fit(x_poly_train, y_train)

# Making predictions
y_predict = lm.predict(x_poly_test)

# Predict CO2 emission for the year 2050
year_value = poly.transform(np.array([[2050]]))
predicted_co2 = lm.predict(year_value)
print("Predicted CO2 emission for 2050:", predicted_co2[0][0])

# Visualizing the polynomial regression curve
x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range_pred = lm.predict(x_range_poly)
print("r2score of the module=",r2_score(y_test,y_predict))
plt.scatter(x, y, label="Scatter Data")
plt.plot(x_range, y_range_pred, color="red", label="Polynomial Regression Curve")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions")
plt.title("Annual Carbon Emissions of China (Polynomial Regression)")
plt.legend()
plt.show()
