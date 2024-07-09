from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
from datetime import timedelta
from datetime import date
import datetime
import pandas as pd 

import pickle

# load the model from disk
loaded_model=pickle.load(open('random_forest_regression_model.pkl', 'rb'))
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('real_2018.csv')
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    date_now = date.today()
    return render_template('result.html',prediction = my_prediction,now=datetime.datetime.now())



if __name__ == '__main__':
	app.run(debug=True)