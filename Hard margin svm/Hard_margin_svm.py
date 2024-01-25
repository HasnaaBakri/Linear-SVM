import numpy as np
import pandas as pd 
import os
os.system("cls")

class HardMarginSVM:
    def __init__(self , learning_rate =.001 , iterations=1000):
        self.learning_rate=learning_rate
        self.iterations=iterations
        self.w=None
        self.b=None
        self.alpha=None

    def gradient_asccent(self , X , y):
        num_samples , num_features=X.shape
        self.alpha=np.zeros(num_samples)
        y=y.reshape(-1 , 1)
        for i in range(self.iterations):
            H=y.dot(y.T) * (X.dot(X.T))
            self.alpha+=self.learning_rate*((np.ones(num_samples) -H.dot( self.alpha)))
        self.alpha=np.where(self.alpha<0 , 0  , self.alpha)
    

    def fit( self, X , y):
        num_samples , num_features=X.shape
        self.w=np.zeros(num_features)
        self.b=0
        self.gradient_asccent(X , y)
        indicies=[i for i in range (num_samples) if self.alpha[i]>0]
        for i in indicies :
            self.w+=self.alpha[i]*y[i]*X[i]
        for i in indicies :
            self.b+=y[i]-np.dot(self.w.T ,X[i] )
        self.b/=len(indicies)

    def predict(self  , X):
        hyperPlane=X.dot(self.w)+self.b
        result=np.where(hyperPlane>=0 , 1, -1)
        return result
    





        

    


