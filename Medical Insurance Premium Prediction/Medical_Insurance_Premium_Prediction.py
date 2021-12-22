# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:41:46 2021

@author: JasonJhan
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# In[1] read data and find the missing value
df = pd.read_csv("Medicalpremium.csv")
print(df.isnull().sum()) # No missing value in this dataset

# In[2] EDA in some continuous colume

labelencoder = LabelEncoder()
df.PremiumPrice = labelencoder.fit_transform(df.PremiumPrice)


df_x = df.drop(columns = ["PremiumPrice"])
xtr,xte,ytr,yte=train_test_split(df_x,df.PremiumPrice,random_state=33,test_size=0.3)
reg=RandomForestRegressor(n_jobs=-1,verbose=2)
param_grid={'n_estimators':[60,50,55],'criterion':['mse','mae'],'max_depth':[7],'min_samples_split':[3],'max_features':['auto']}
gs=GridSearchCV(reg,param_grid=param_grid,cv=3,n_jobs=-1,verbose=1)
gs.fit(xtr,ytr)
