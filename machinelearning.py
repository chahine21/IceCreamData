import pandas as pd
import numpy as np
import pickle   
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 

df=pd.read_csv("IceCreamData.csv")

X=df['Temperature'] 
y=df['Revenue'] 

X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2)
model= LinearRegression() 

np.array([X_train]).ndim 

model.fit(np.array([X_train]).reshape(-1,1),y_train) 

y_pred=model.predict(np.array([X_test]).reshape(-1,1)) 

print(model.predict([[1]]))
with open ('model.pkl','wb') as files:
    pickle.dump(model,files )
