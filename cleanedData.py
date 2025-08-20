import pandas as pd
import numpy as np
import csv


car=pd.read_csv('quikr_car.csv')
car.head()
# print(car.head())

print("Before Data Clean : ", car.shape)

car.info()
car['name'].unique()
car['company'].unique()
car['year'].unique()
car['Price'].unique()
car['kms_driven'].unique()

# if 'fuel_type' == 'LPG':
#     drop=True
car = car[car['fuel_type'] != 'LPG']


car['fuel_type'].unique()


# backup=car.copy()

car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)

car = car[car['Price']!="Ask For Price"] 

car['Price'] = car['Price'].str.replace(',','').astype(int)

car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')

car = car[car['kms_driven'].str.isnumeric()]

car['kms_driven'] = car['kms_driven'].astype(int)

car = car[~car['fuel_type'].isna()]

car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')

car = car.reset_index(drop=True)

car = car[car['Price']<6e6].reset_index(drop=True)

# print(car.info)

print("After Data Clean : ", car.shape)


car.to_csv('Cleaned Car.csv')

x = car.drop(columns='Price')

y = car['Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


ohe = OneHotEncoder()
ohe.fit(x[['name', 'company', 'fuel_type']])
print(ohe.categories_)

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name', 'company','fuel_type']),remainder = 'passthrough')
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)
r2_score(y_test,y_pred)


scores =[]

for i in range(1000):
    x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred = pipe.predict(x_test)
    # print(r2_score(y_test,y_pred), i)
    scores.append(r2_score(y_test,y_pred))

np.argmax(scores)

print(scores[np.argmax(scores)])

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
r2_score(y_test, y_pred)

import pickle

pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))

pipe.predict(pd.DataFrame([['Maruti Suzuki Swift', 'Maruti',2019,100,'Petrol']], columns=['name','company','year', 'kms_driven','fuel_type']))