# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:24:46 2020

@author: Administrator
"""
#python -m pip install xgboost
'''
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
class_1=500
class_2=500
centers=[[0.0, 0.0], [2.0, 2.0]]
clusters_std=[0.5,0.5]
x1,y1=make_blobs(n_samples=500,centers=[[0,0]],cluster_std=0.5,random_state=0,shuffle=False)
x2,y2=make_blobs(n_samples=500,centers=[[2,2]],cluster_std=0.5,random_state=0,shuffle=False)
x1=pd.DataFrame(x1)
x2=pd.DataFrame(x2)
y1=pd.DataFrame(y1)
y2=pd.DataFrame(y2)
y2[:]=1
x=pd.concat([x1,x2])
y=pd.concat([y1,y2])
x=np.array(x)
y=np.array(y)
'''
'''
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
cv=KFold(n_splits=10,shuffle=True,random_state=42)
train_sizes,train_scores,test_scores=learning_curve(XGBR(n_estimators=100,random_state=420),x,y,shuffle=True,cv=cv)
plt.plot(train_sizes,np.mean(train_scores,axis=1),'o-',color="r",label="Training score")
plt.plot(train_sizes,np.mean(test_scores,axis=1),'o-',color="g",label="Test score") 
plt.show()
'''


import numpy as np
import pandas as pd
f=open('D:\\毕设\\姚\\colon-cancer.txt')
d=pd.read_table(f,sep=' ',header=None)
data=d.iloc[:,2:]
data_=data
for i in range(62):
    for j in range(2000):
        data_.iloc[i,j]=float(data.iloc[i,j].split(':')[1])












