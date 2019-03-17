# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:48:59 2019

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Data.csv')
"Independent variables"
X = dataset.iloc[:,:-1].values
"Dependent variables"
Y = dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean' , axis =0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])
