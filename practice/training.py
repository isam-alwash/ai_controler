import numpy as np

IP = np.array([[ 0 , 0 ],
               [ 0 , 1 ],
               [ 1 , 0 ],
               [ 1 , 1 ]])

OP = np.array([ [0] , [0] , [0] , [1] ])
np.random.seed(1)
w = np.random.rand( 2 , 1 ) 
b = np.random.rand(1)
r = 0.1

def sigmoid( z ):
    return 1 / (1 + np.exp(-z)) 

def Dsigmoid( x ):
    return x * ( 1 - x)
while 1:
    POP = sigmoid( np.dot(IP , w) + b)
    e = OP - POP
    adjust = e * Dsigmoid(POP)
    b += r * np.sum(adjust)
    w += r * np.dot(IP.T , adjust)
    print(POP)
    print(" ")
    print(np.round(POP))