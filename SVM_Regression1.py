# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:40:16 2018
@author: Chirag mahawar"""

#Machine Learning to predict Google Play Store Application Rating

#Step 1 - Data Preprocessing

#importing the libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('googleplaystore.csv')

#remove missing values from dataset
dataset.dropna(inplace = True)

#dropping of unrelated and unnecessary columns from the dataset 
dataset.drop(labels = ['Last Updated','Current Ver','Android Ver','App','Genres'], axis = 1, inplace = True)

# Cleaning Categories into integers
CategoryString = dataset["Category"]
categoryVal = dataset["Category"].unique()
categoryValCount = len(categoryVal)
category_dict = {}
for i in range(0,categoryValCount):
    category_dict[categoryVal[i]] = i
dataset["Category_c"] = dataset["Category"].map(category_dict).astype(int)

#cleaning size of installation
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)*1000000
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)*1000
        return(x)
    else:
        return None

dataset["Size"] = dataset["Size"].map(change_size)
#filling Size which had NA
dataset.Size.fillna(method = 'ffill', inplace = True)

#Cleaning no of installs column
dataset['Installs'] = [int(i[:-1].replace(',','')) for i in dataset['Installs']]

#Converting Type column into binary column
def type_cat(types):
    if types == 'Free':
        return 0
    else:
        return 1

dataset['Type'] = dataset['Type'].map(type_cat)

#Cleaning of content rating classification
RatingL = dataset['Content Rating'].unique()
RatingDict = {}
for i in range(len(RatingL)):
    RatingDict[RatingL[i]] = i
dataset['Content Rating'] = dataset['Content Rating'].map(RatingDict).astype(int)

#Cleaning prices
def price_clean(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        price = float(price)
        return price

dataset['Price'] = dataset['Price'].map(price_clean).astype(float)

# convert reviews to numeric
dataset['Reviews'] = dataset['Reviews'].astype(int)

# for dummy variable encoding for Categories
dataset = pd.get_dummies(dataset, columns=['Category'])

"""Step 2 - Training & Testing of the model
Model 2 : SVM Regression(without feature scaling)"""

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 0)

#fitting the SVR model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')   #chhosing the gaussian kernel..ie rbf kernel
regressor.fit(X_train , y_train)

y_pred_svm = regressor.predict(X_test)

def Evaluationmatrix(y_test, y_pred):
    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_test,y_pred)))
    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_test,y_pred)))
    
Evaluationmatrix(y_test,y_pred_svm)

