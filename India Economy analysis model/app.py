from flask import Flask, render_template,request,jsonify
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def index():
    # Load data
    df = pd.read_excel("India Economy.xlsx", header=None)

    
    x = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 2].values.reshape(-1, 1)
    manufacture = df[4]
    imported = df[6]
    export = df[8]
    inflation = df[10]
    population = df[12]


    # Polynomial Regression
    poly = PolynomialFeatures(degree=3)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)

    # Sort for smooth plot
    sorted_idx = x[:, 0].argsort()
    x_sorted = x[sorted_idx]
    y_sorted = y_pred[sorted_idx]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue', label='Actual GDP')
    plt.plot(x_sorted, y_sorted, color='red', label='Polynomial Fit')
    plt.xlabel('Year')
    plt.ylabel('GDP Growth')
    plt.title('India GDP Polynomial Regression')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    x=df[0]
    # Save plot
    
    plot_path = os.path.join("static", "plot.png")
    plt.savefig(plot_path)
    plt.close()

    plt.plot(x, manufacture, linestyle='-')
    plt.title('Manufacturing Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Manufacture (Billion)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_manu=os.path.join("static","manu.png")
    plt.savefig(plot_manu)
    plt.close()
    plt.bar(x, imported, color='orange', label='Imports')

    plt.title('Products Imported from Other Countries')
    plt.xlabel('Year')
    plt.ylabel('Imports (Billion)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_impo=os.path.join("static","impo.png")
    plt.savefig(plot_impo)
    plt.close()

    plt.bar(x, export, color='green')
    plt.title("Products Exported from Our Country")
    plt.xlabel('Year')
    plt.ylabel('Exports (Billion)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_expo=os.path.join("static","expo.png")
    plt.savefig(plot_expo)
    plt.close()

    plt.plot(x,inflation)
    plt.title("Inflation Rate in India")
    plt.ylabel('Inflation Rate (%)')
    plt.xlabel('year')

    plot_infla=os.path.join("static","infla.png")
    plt.savefig(plot_infla)
    plt.close()

    year=x.values.reshape(-1,1)
    population=population.values.reshape(-1,1)
    x_train,x_test,y_train,y_test=train_test_split(year,population,train_size=0.33,random_state=47)
    lm=LinearRegression()
    lm.fit(x_train,y_train)
    y_pred=lm.predict(x_test)

    
    plt.scatter(year,population)
    plt.plot(x_test,y_pred,color='black')
    plot_pop=os.path.join("static","pop.png")
    plt.savefig(plot_pop)
    plt.close()

    return render_template("index.html",
    plot_gdp="plot.png",
    plot_manu="manu.png",
    plot_impo="impo.png",
    plot_expo="expo.png",
    plot_infla="infla.png",
    plot_pop="pop.png"
    )

from flask import Flask, render_template, request, jsonify
# Keep existing imports...

@app.route('/pred',methods=['GET','POST'])
def prediction():
    year=request.form.get('year')
    if year==None:
        return render_template('pred.html')
    else:
        year=int(year)
        print(year)
        df = pd.read_excel("India Economy.xlsx", header=None)
        x = df.iloc[:, 0].values.reshape(-1, 1)
        y = df.iloc[:, 2].values.reshape(-1, 1)

            # Polynomial regression model
        poly = PolynomialFeatures(degree=3)
        x_poly = poly.fit_transform(x)
        model = LinearRegression()
        model.fit(x_poly, y)

            # Predict for the input year
        year_input = np.array([[year]])
        year_input_poly = poly.transform(year_input)
        predicted_gdp = model.predict(year_input_poly)[0][0]
        predicted_gdp=round(predicted_gdp)
        print(predicted_gdp)
    
        population = df.iloc[:, 12].values.reshape(-1, 1)
        pop_model = LinearRegression()
        pop_model.fit(x, population)
        predicted_pop = pop_model.predict(np.array([[year]]))[0][0]
        predicted_pop=round(predicted_pop)
        # formatted_pop = f"{predicted_pop:,.0f}
        return render_template('pred.html',gdp=predicted_gdp,pop=predicted_pop)
    

if __name__ == "__main__":
    app.run(debug=True)
