import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib


def read_in():
    #read in
    pollutant = pd.read_csv('pollutant_merge_polished.csv')
    return pollutant


def target_and_split(pollutant):
    #trains and returns the model
    x = pollutant[['temperature']]
    y = pollutant[['pm2.5']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 1)

    lr_model = LinearRegression()

    lr_model.fit(x_train, y_train)

    joblib.dump(lr_model, 'data/lr_model.pkl')

    return x_test, y_test, lr_model


