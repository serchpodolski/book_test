# Create Flask app to deploy the trained Linear Regression model into production
# Load dependencies
import pandas as pd
import numpy as np
import sklearn
import joblib
from flask import Flask, render_template, request

# Create the Flask app and assign it to a variable, would need to add a static folder if a external CSS file usage is intended
app = Flask(__name__)

# Routing for home page, this decorator will run the function to display the HTML landing site
@app.route('/')
def home():
    return render_template('home.html')

# Create the route that will implement the predict method from the LR model we trained in our notebook
@app.route('/predict', methods=['GET', 'POST'])
def predict():
     if request.method == 'POST':
          print(request.form.get('var_1'))
          print(request.form.get('var_2'))
          print(request.form.get('var_3'))
          print(request.form.get('var_4'))
          print(request.form.get('var_5'))
          try:
               var_1=float(request.form['var_1'])
               var_2=float(request.form['var_2'])
               var_3=float(request.form['var_3'])
               var_4=float(request.form['var_4'])
               var_5=float(request.form['var_5'])
               pred_args=[var_1,var_2,var_3,var_4,var_5]
               pred_arr=np.array(pred_args)
               preds=pred_arr.reshape(1,-1)
               model = open('linear_regression_model.pkl', 'rb')
               lr_model=joblib.load(model)
               model_prediction=lr_model.predict(preds)
               model_prediction=round(float(model_prediction),2)
          except ValueError:
               return "Please enter valid values"
     return render_template('predict.html', prediction=model_prediction)

if __name__ == '__main__':
     app.run(host='0.0.0.0', debug=True)