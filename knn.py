# import required libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
Ids = test.Id.values
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)

# combine common features for processing together
combined = train.drop('SalePrice', axis = 1).append(test)
combined.drop('Id', axis = 1, inplace = True)

lessMissing = combined.isnull().sum() * 100 / combined.shape[0] < 15
combined = combined.loc[:, lessMissing]

combined['GarageCars'].fillna(combined['GarageCars'].mode().values[0], inplace = True)
combined['MSZoning'].fillna(combined['MSZoning'].mode().values[0], inplace = True)
combined['Utilities'].fillna(combined['Utilities'].mode().values[0], inplace = True)
combined['BsmtFinSF2'].fillna(combined['BsmtFinSF2'].mean(), inplace = True)
combined['TotalBsmtSF'].fillna(combined['TotalBsmtSF'].mean(), inplace = True)
combined['BsmtHalfBath'].fillna(combined['BsmtHalfBath'].mode().values[0], inplace = True)
combined['GarageArea'].fillna(np.floor(combined['GarageArea'].mean()), inplace = True)
combined['SaleType'].fillna(combined['SaleType'].mode()[0], inplace = True)
combined['Functional'].fillna(combined['Functional'].mode()[0], inplace = True)
combined['BsmtFinSF1'].fillna(combined['BsmtFinSF1'].mean(), inplace = True)
combined['BsmtUnfSF'].fillna(combined['BsmtUnfSF'].mean(), inplace = True)
combined['Electrical'].fillna(combined['Electrical'].mode()[0], inplace = True)
combined['KitchenQual'].fillna('TA', inplace = True)
combined['GarageCond'].fillna(combined['GarageCond'].mode()[0], inplace = True)
combined['GarageQual'].fillna(combined['GarageQual'].mode()[0], inplace = True)
combined['GarageFinish'].fillna(combined['GarageFinish'].mode()[0], inplace = True)
combined['GarageType'].fillna(combined['GarageType'].mode()[0], inplace = True)
combined['BsmtFullBath'].fillna(1.0, inplace = True)
combined['BsmtFinType2'].fillna(combined['BsmtFinType2'].mode()[0], inplace = True)
combined['GarageYrBlt'].fillna(combined['GarageYrBlt'].mode()[0], inplace = True)
combined['MasVnrType'].fillna(combined['MasVnrType'].mode()[0], inplace = True)
combined['MasVnrArea'].fillna(combined['MasVnrArea'].mean(), inplace = True)
combined['Exterior1st'].fillna(combined['Exterior1st'].mode()[0], inplace = True)
combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode()[0], inplace = True)
combined['BsmtCond'].fillna(combined['BsmtCond'].mode()[0], inplace = True)
combined['BsmtExposure'].fillna(combined['BsmtExposure'].mode()[0], inplace = True)
combined['BsmtFinType1'].fillna('GLQ', inplace = True)
combined['BsmtQual'].fillna('Gd', inplace = True)

significant = ['LotArea', 'OverallQual', 'YearBuilt', 'BsmtQual', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'KitchenQual' ,'GarageType', 'ScreenPorch']
combined = combined.loc[:, significant]

# convert some columns to categorical
nonCategorical = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']
categorical = []
minMaxScaler = preprocessing.MinMaxScaler()
for column in combined.columns:
    if column in nonCategorical:
        combined[column] = combined[column].astype('float64')
        combined[column] = minMaxScaler.fit_transform(combined[[column]])
    else:
        combined[column] = combined[column].astype('category')
        categorical = np.append(categorical, column)

combined = pd.get_dummies(combined, columns = categorical)
redundant = ['OverallQual_1', 'YearBuilt_1872', 'BsmtQual_Ex', 'KitchenQual_Ex', 'GarageType_2Types']
combined.drop(redundant, axis = 1, inplace = True)

# split into train and test again
X = combined.iloc[0:train.shape[0], :]
X['SalePrice'] = train['SalePrice']
xTest = combined.iloc[train.shape[0]:combined.shape[0], :]

training = X.copy()
testing = xTest.copy()

def knn(k, trainData = X, testData = xTest):
    predictions = np.array([])
    for test in range(0, len(testData)):
        distances = np.array([])
        for train in range(0, len(trainData)):
            distance = math.sqrt(np.square(xTest.iloc[test, :] - X.iloc[train, :]).sum())
            distances = np.append(distances, distance)
        training['Distances'] = 1 / distances
        closest = training.sort_values(by = ['Distances'], ascending = False).head(k)
        weights = closest['Distances']
        prices = closest['SalePrice']
        prediction = (weights * prices).sum() / weights.sum()
        predictions = np.append(predictions, prediction)
    return predictions

predictions = knn(5)

submission = pd.DataFrame({'Id': Ids, 'SalePrice': predictions})
submission.to_csv('submissionKNN.csv', index = False)