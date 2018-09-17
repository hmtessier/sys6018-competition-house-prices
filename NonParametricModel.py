# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:17:41 2018
@author: Henry
"""

import pandas as pd
import numpy as np
import operator

# Read in data to dataframes
df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Code all categorical variables for train (df) and test frames
cols = df.columns[df.dtypes.eq('object')]
for col in df[cols]:
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.codes
df = df.fillna(df.mean())

for col in test[cols]:
    test[col] = test[col].astype('category')
    test[col] = test[col].cat.codes
test = test.fillna(test.mean())

# Function to standardize data
def standardize(df):
    standard = df.copy()
    for col_name in df.columns: 
        max_val = max(df[col_name])
        min_val = min(df[col_name])
        standard[col_name] = (df[col_name]- min_val)/(max_val-min_val)
    return standard

# Select only relevant columns for each dataframe
Ids = test.Id
test = test.iloc[:,1:]
SalePr = df.SalePrice
df = df.iloc[:,1:]

# Standardize the data in each dataframe (except response variable)
df = standardize(df)
test = standardize(test)
df['SalePrice'] = SalePr

# Function for Euclidean Distance
def euclideanDist(obj1, obj2, length):
    dist = 0
    for n in range(length):
        dist += ((obj1[n] - obj2[n])**2)
    return dist**.5

# Function for finding the k nearest neighbors using Euclidean Distance function
def nearestNeighbors(train, testcase, k):
    distances = []
    kclosest = []
    nvec = len(testcase)
    for n in range(len(train)):
        dist = euclideanDist(testcase, train[n], nvec)
        distances.append((train[n],dist))
    distances.sort(key=operator.itemgetter(1))
    for n in range(k):
        kclosest.append(distances[n])
    return kclosest

# Prep dataframes for entry into functions
test = test.values.tolist()
df = df.values.tolist()

# Predict values for each testcase by calulating the inverse weighted average for 10 nearest neighbors
pred = []
for n in range(0,len(test)):
    ks = nearestNeighbors(df, test[n],10)
    prices = []
    weights = []
    for x in range(0,10):
        price = ks[x][0][79]
        weight = 1/(ks[x][1] + .00000000000000001) 
        # Add small value on off-chance that distance between two vectors is zero (can't divide by zero)
        prices.append(price)
        weights.append(weight)
    weights = [i /sum(weights) for i in weights]
    total = [a*b for a,b in zip(weights,prices)]
    avg = sum(total)
    pred.append(avg)
    print(n)    

# Output predictions to dataframe and excel  
out = pd.DataFrame(columns = ['Id','SalePrice'])
out['Id'] = Ids
out['SalePrice'] = pred
out.to_csv('NonParametric.csv',index=False)




        