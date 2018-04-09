# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:39:37 2018

@author: Brian Nguyen
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#Read in datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#View correlations
# =============================================================================
# corr = train.corr()
# plt.figure(figsize = (20,20))
# sns.heatmap(corr, 
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# =============================================================================

#Subset by features and labels
x_train = np.array(train.loc[:,["OverallQual", "LotArea", "GrLivArea"]])
y_train = train.iloc[:,-1]

x_test = np.array(test.loc[:,["OverallQual", "LotArea", "GrLivArea"]])
y_test = test.iloc[:,-1]

#Create a split for training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

#Normalize the data
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)

scaler.fit(x_test)
x_val = scaler.transform(x_val)

#Create random forest model
model = RandomForestRegressor()

model.fit(x_train, y_train)
y_pred = model.predict(x_val)
print(model.score(x_val, y_val))

#Scatter plot to see similiarities
plt.scatter(range(len(y_pred)), y_pred)
plt.scatter(range(len(y_val)), y_val)