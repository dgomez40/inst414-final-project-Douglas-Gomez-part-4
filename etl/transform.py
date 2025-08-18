#load in merge file
from sklearn.preprocessing import StandardScaler
import etl.extract as extract
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression





# def unique_id():
#     #gives unique id
#     pollutant_merge = etl.merge()
#     pollutant_merge = pollutant_merge.copy()
#     pollutant_merge['id'] = range(1, len(pollutant_merge) + 1)
#     # moves 'id' column to the front
#     cols = ['id'] + [col for col in pollutant_merge.columns if col != 'id']
#     pollutant_merge = pollutant_merge[cols]
#     return pollutant_merge
#     #NO LONGER NEEDED

# def missing_value(pollutant_merge):
#     #drops rows with missing values
#     pollutant_merge = etl.merge()
#     pollutant_merge = pollutant_merge.copy()
#     pollutant_merge - pollutant_merge.dropna()
#     return pollutant_merge
# #not good idea because some observations were not taken leading to msising values and might drop a ton of information

def drop_columns():
    pollutant_merge = extract.merge()
    keep_cols = [
    'DATE', 'NAME',
    'PRCP',  # Precipitation
    'SNOW', 'SNWD',  # Snow data
    'TMAX', 'TMIN', 'TAVG',  # Temperatures
    'AWND',  # Average wind speed
    'State Name', 'county Name', 'State Code', 'County Code',
    'AQI', 'Category'
]
    pollutant_merge = pollutant_merge[keep_cols]
    # print(pollutant_merge)
    pollutant_merge.to_csv('data/pollutant_merge.csv')


#since I am doing linear regression, I think it would be best to remove outliers from the dataset because it would help with training the model

def outliers():
    #load data
    pollutant_merge = pd.read_csv("data/pollutant_merge.csv")
    # print(pollutant_merge)
    
    #drops outliers
    # q1 = pollutant_merge['pm2.5'].quantile(0.25)
    # q3 = pollutant_merge['pm2.5'].quantile(0.75)
    # iqr = q3 - q1
    # pollutant_merge = pollutant_merge[(pollutant_merge['pm2.5'] >= q1 - 1.5 * iqr) & (pollutant_merge['pm2.5'] <= q3 + 1.5 * iqr)]
    # return pollutant_merge
    Q1 = pollutant_merge['TAVG'].quantile(0.25)
    Q3 = pollutant_merge['TAVG'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    pollutant_merge =  pollutant_merge[(pollutant_merge['TAVG'] >= lower) & (pollutant_merge['TAVG'] <= upper)]
    pollutant_merge.to_csv('data/pollutant_merge.csv')
    #removes 2 rows basically


#since linear regression is sensitive to feature scale, I am also going to normalize the data as well.
def split_into_train_and_test():
    pollutant_merge = pd.read_csv("data/pollutant_merge.csv")
    column = ['TAVG']
    target_column = ['AQI']
    X = pollutant_merge[column]
    y = pollutant_merge[target_column]
    return X, y

def split_train_test_data(X, y, test_size=0.2, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    train_X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    test_X_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return train_X_scaled, test_X_scaled, scaler

def train_lin_reg():
    X, y = split_into_train_and_test()
    X_train, X_test, y_train, y_test = split_train_test_data(X,y)
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results = pd.DataFrame({
    'Actual_AQI': y_test.values.ravel(), 
    'Predicted_AQI': y_pred.ravel()     
})
    # print(results)
    return model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred


    















