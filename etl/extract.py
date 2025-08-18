import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np


def air_pollutant():

    pm25_raw = pd.read_csv(
        "data/pm25.csv",
        parse_dates=['DATE'],
        date_format= '%m/%d/%y'

    )
    pm25_raw.to_csv("data/pm25_raw.csv")
    return pm25_raw


def weather():

    temperature_raw = pd.read_csv(
    'data/temperature.csv',
    parse_dates=['DATE'],
    date_format='%m/%d/%y'
)
    # print(temperature_raw)
    temperature_raw.to_csv('data/temperature_raw.csv')
    return temperature_raw


def merge():
    # Load datasets
    temperature_raw = weather()
    pm25_raw = air_pollutant()
    
    # Find the latest date in the smaller dataset
    max_pm25_date = pm25_raw['DATE'].max()
    # print(max_pm25_date)

    # # Filter temperature_raw so it only goes up to the max date (Oct 31 in this case)
    temperature_filtered = temperature_raw[temperature_raw['DATE'] <= max_pm25_date]
    # print(temperature_filtered)

    # # Merge on the date column
    pollutant_merge = pd.merge(temperature_filtered, pm25_raw, on='DATE')
    pollutant_merge.to_csv('data/pollutant_merge.csv')

    return pollutant_merge