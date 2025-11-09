import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
from math import cos, asin, sqrt, pi
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("uber.csv")
df.drop(columns=['Unnamed: 0', 'key'], inplace=True)
df.dropna(how='any', inplace=True)

for col in df.select_dtypes(exclude=['object']):
    plt.figure()
    sns.boxplot(data=df, x=col)

df = df[
    (df.pickup_latitude > -90) & (df.pickup_latitude < 90) &
    (df.dropoff_latitude > -90) & (df.dropoff_latitude < 90) &
    (df.pickup_longitude > -180) & (df.pickup_longitude < 180) &
    (df.dropoff_longitude > -180) & (df.dropoff_longitude < 180) &
    (df.fare_amount > 0) &
    (df.passenger_count > 0) &
    (df.passenger_count < 50)
]

def distance(lat_1, lon_1, lat_2, lon_2):
    lon_1, lon_2, lat_1, lat_2 = map(np.radians, [lon_1, lon_2, lat_1, lat_2])
    diff_lon = lon_2 - lon_1
    diff_lat = lat_2 - lat_1
    km = 2 * 6371 * np.arcsin(np.sqrt(np.sin(diff_lat / 2.0) ** 2 +
           np.cos(lat_1) * np.cos(lat_2) * np.sin(diff_lon / 2.0) ** 2))
    return km

temp = distance(df['pickup_latitude'], df['pickup_longitude'],
                df['dropoff_latitude'], df['dropoff_longitude'])

df['Distance'] = temp
sns.boxplot(data=df, x='Distance')

df = df[(df['Distance'] < 200) & (df['Distance'] > 0)]
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['week_day'] = df['pickup_datetime'].dt.day_name()
df['Year'] = df['pickup_datetime'].dt.year
df['Month'] = df['pickup_datetime'].dt.month
df['Hour'] = df['pickup_datetime'].dt.hour

df.drop(columns=['pickup_datetime', 'pickup_latitude', 'pickup_longitude',
                 'dropoff_latitude', 'dropoff_longitude'], inplace=True)

temp = df.copy()

def convert_week_day(day):
    if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday']:
        return 0
    return 1

def convert_hour(hour):
    if 5 <= hour <= 12:
        return 1
    elif 12 < hour <= 17:
        return 2
    elif 17 < hour < 24:
        return 3
    return 0

df['week_day'] = temp['week_day'].apply(convert_week_day)
df['Hour'] = temp['Hour'].apply(convert_hour)

sns.scatterplot(y=df['fare_amount'], x=df['Distance'])

x = df[['Distance']].values
y = df['fare_amount'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.transform(x_test)

std_y = StandardScaler()
y_train = std_y.fit_transform(y_train)
y_test = std_y.transform(y_test)

def fit_predict(model):
    model.fit(x_train, y_train.ravel())
    y_pred = model.predict(x_test)
    r_squared = r2_score(y_test, y_pred)
    RMSE = mean_squared_error(y_test, y_pred, squared=False)
    MAE = mean_absolute_error(y_test, y_pred)
    print('R-squared: ', r_squared)
    print('RMSE: ', RMSE)
    print("MAE: ", MAE)

fit_predict(LinearRegression())
fit_predict(RandomForestRegressor())
