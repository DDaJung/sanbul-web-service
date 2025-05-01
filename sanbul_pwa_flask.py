import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

import numpy as np
import pandas as pd
from flask import Flask, render_template

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
Bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = pd.DataFrame([{
            'latitude': float(form.latitude.data),
            'longitude': float(form.longitude.data),
            'month': form.month.data,
            'day': form.day.data,
            'avg_temp': float(form.avg_temp.data),
            'max_temp': float(form.max_temp.data),
            'max_wind_speed': float(form.max_wind_speed.data),
            'avg_wind': float(form.avg_wind.data)
        }])

        print(X_test.shape)
        print(X_test)

        data = pd.read_csv('./sanbul-5.csv', sep=',')

        num_attribs = ['latitude', 'longitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
        cat_attribs = ['month', 'day']

        X = data[num_attribs + cat_attribs]
        y = data["burned_area"]

        num_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        full_pipeline.fit(X)

        X_test = full_pipeline.transform(X_test)

        model = keras.models.load_model('fires_model.keras')

        prediction = model.predict(X_test)
        res = prediction[0][0]
        res = np.round(res * 100, 2)
        res = float(res)

        return render_template('result.html', res=res)
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()