# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:17:41 2018

@author: Henry
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#code all categorical variables
cols = df.columns[df.dtypes.eq('object')]
for col in df[cols]:
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.codes
df = df.fillna(df.mean())

for col in test[cols]:
    test[col] = test[col].astype('category')
    test[col] = test[col].cat.codes
test = test.fillna(test.mean())

#create initial model with all variables
model1 = sm.OLS(df.SalePrice, df.iloc[:,1:80]).fit()

#potential vars to remove: 
#LotConfig, YearRemodAdd, Foundation, CentralAir, BsmtHalfBath, HalfBath, GarageCond, EnclosedPorch, Fence, MiscVal, MoSold 

#create list of only significant variables
significants = np.setdiff1d(df.iloc[:,1:80].columns, ['LotConfig','YearRemodAdd','Foundation','CentralAir','BsmtHalfBath','HalfBath','GarageCond','EnclosedPorch','Fence','MiscVal','MoSold'])

#create new model using only variables from list
model2 = sm.OLS(df.SalePrice, df[significants]).fit()

#predict using new model
pred = model2.predict(test[significants])

#output to df and csv
out = pd.DataFrame(columns = ['Id','SalePrice'])
out['Id'] = test.Id
out['SalePrice'] = pred
out.to_csv('ModelOutput.csv',index=False)
