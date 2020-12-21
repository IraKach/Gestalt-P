# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 09:52:36 2020

@author: user
"""
# Importing the libraries
import numpy as np
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('C:\\Users\\user\\autos.csv')



# Making a list of missing value types
#print (dataset.isnull().sum())
missing_values = ["?"]
dataset = pd.read_csv("autos.csv", na_values = missing_values)

X = dataset.iloc[:, :-1].values 
#Y = df.iloc[:, 24].values 
Y = dataset.iloc[:, [24]].values
from sklearn.impute import SimpleImputer
#handling missing Data in column 1,17,18,20,21
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
X[:,[1,17,18,20,21]]=imp_median.fit_transform(X[:,[1,17,18,20,21]])
#print (X[:,[1,17,18,20,21]])

#handling missing Data in column 4
imp_median = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[:,[4]]=imp_median.fit_transform(X[:,[4]])
#print (X[:,[4]])
#handling missing Data in last column, y
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
Y[:]=imp_median.fit_transform(Y[:])
#print (Y[:])

#handling the categorical feature
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder = LabelEncoder()
X[:, 2] = labelencoder.fit_transform(X[:, 2])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder = LabelEncoder()
X[:, 2] = labelencoder.fit_transform(X[:, 2])
labelencoder2 = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
labelencoder3 = LabelEncoder()
X[:, 4] = labelencoder.fit_transform(X[:, 4])
labelencoder4 = LabelEncoder()
X[:, 5] = labelencoder.fit_transform(X[:, 5])
labelencoder5 = LabelEncoder()
X[:, 6] = labelencoder.fit_transform(X[:, 6])
labelencoder6 = LabelEncoder()
X[:, 7] = labelencoder.fit_transform(X[:, 7])
labelencoder7 = LabelEncoder()
X[:, 13] = labelencoder.fit_transform(X[:, 13])
labelencoder8 = LabelEncoder()
X[:, 14] = labelencoder.fit_transform(X[:, 14])
labelencoder9 = LabelEncoder()
X[:, 16] = labelencoder.fit_transform(X[:, 16])

onehotencoder = OneHotEncoder(sparse=False)
Z= onehotencoder.fit_transform(X[:, [5]])
X = np.hstack(( Z, X[:,:5] , X[:,6:])).astype('float')
#handling the dummy variable trap
X = X[:, 1:]
onehotencoder = OneHotEncoder(sparse=False)
Z1= onehotencoder.fit_transform(X[:, [9]])
X = np.hstack(( Z1, X[:,:9] , X[:,10:])).astype('float')
#handling the dummy variable trap
X = X[:, 1:]
onehotencoder = OneHotEncoder(sparse=False)
Z2= onehotencoder.fit_transform(X[:, [18]])
X = np.hstack(( Z2, X[:,:18] , X[:,19:])).astype('float')
#handling the dummy variable trap
X = X[:, 1:]
onehotencoder = OneHotEncoder(sparse=False)
Z3= onehotencoder.fit_transform(X[:, [25]])
X = np.hstack(( Z3, X[:,:25] , X[:,26:])).astype('float')
#handling the dummy variable trap
X = X[:, 1:]
import statsmodels.tools.tools as tl
X = tl.add_constant(X)
import statsmodels.api as sm
def forward_selection(x):
    
    vars_number = len(X[0])
    #print (vars_number)
    selectedVars = []
    minSig = 0.05
    while True:
	    min = 1
	    selected = -1
	    for i in range(vars_number):
		    if i in selectedVars:
			    continue
		    tmpSelectedVars = selectedVars[:]
		    tmpSelectedVars.append(i)
		    x = X[:,tmpSelectedVars]
		    
		    regressor_OLS=sm.OLS(endog=Y, exog=x).fit()
		    pvalue = regressor_OLS.pvalues[-1]
		    if pvalue < min and pvalue < minSig:
			    min = pvalue
			    selected = i

	    if selected == -1:
		    break
	    else:
		    selectedVars.append(selected)
    print (selectedVars)

SL = 0.05
X_opt=X[ :, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]]
print("***")
X_Modeled = forward_selection(X_opt)

X= X[:, [32, 9, 15, 25, 17, 23, 13]]
#split to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
from sklearn.tree import DecisionTreeRegressor
regressor_2 = DecisionTreeRegressor(max_depth=3, random_state=0)
regressor_2.fit(X_train, y_train)
y_predTree = regressor_2.predict(X_train)
residualsaverage2=np.average(np.abs(y_predTree-y_train))
print(residualsaverage2)





