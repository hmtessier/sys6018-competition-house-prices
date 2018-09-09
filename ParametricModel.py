# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:17:41 2018

@author: Henry
"""

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

cols = df.columns[df.dtypes.eq('object')]
for col in df[cols]:
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.codes
df = df.fillna(df.mean())

for col in test[cols]:
    test[col] = test[col].astype('category')
    test[col] = test[col].cat.codes
test = test.fillna(test.mean())


model = sm.OLS(df.SalePrice, df.iloc[:,1:80]).fit()

pred = model.predict(test.iloc[:,1:])

out = pd.DataFrame(columns = ['Id','SalePrice'])
out['Id'] = test.Id
out['SalePrice'] = pred

out.to_csv('ModelOutput.csv',index=False)