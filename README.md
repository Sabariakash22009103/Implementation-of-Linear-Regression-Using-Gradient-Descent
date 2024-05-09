<h3>Date:</h3>
<h2>Exp: 03</h2>

# Implementation of Linear Regression Using Gradient Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Initialize weights randomly.
2. Compute predicted values.
3. Compute gradient of loss function.
4. Update weights using gradient descent.
## Program:
Program to implement the linear regression using gradient descent.

Developed by: Sabari Akash A

RegisterNumber:  212222230124

```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

#Assuming the last column is your target variable 'y' and the 
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X)
print(X1_scaled)

#learn model parameters
theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value:{pre}")
```
## Output:
<img src= image.png width=300 height=100>


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
