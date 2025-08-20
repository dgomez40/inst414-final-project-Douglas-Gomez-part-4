import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)

def air_pollutant():
    logger.info("creating air_pollutant")
    url = 'https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_2024.zip'
    pm25_raw = pd.read_csv(url)
    # print(pm25_raw)
    #only get rows in anne arundel
    pm25_raw = pm25_raw[pm25_raw["county Name"] == "Anne Arundel"]
    #change name to upper case and make date datetime format
    pm25_raw.rename(columns={"Date": "DATE"}, inplace=True)
    pm25_raw.to_csv("data/pm25_raw.csv")
    # return pm25_raw


def weather():
    logger.info("creating temperature csv")
    url = 'https://www.ncei.noaa.gov/data/daily-summaries/access/USW00093721.csv'
    temperature_raw = pd.read_csv(url)
    # print(temperature_raw)
    temperature_raw["DATE"] = pd.to_datetime(temperature_raw["DATE"])
    temperature_raw = temperature_raw[temperature_raw["DATE"].dt.year == 2024]
    temperature_raw.to_csv('data/temperature_raw.csv')
    # return temperature_raw


def merge():
    # Load datasets
    temperature_raw = pd.read_csv("data/pm25_raw.csv")
    pm25_raw = pd.read_csv("data/temperature_raw.csv")
    
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