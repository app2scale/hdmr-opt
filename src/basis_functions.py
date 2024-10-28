
import math
import numpy as np

## ------------------ BASIS FUNCTIONS START ------------------ ##
def Pn(m, x):
        if m == 0:
            return np.ones_like(x)
        elif m == 1:
            return x
        else:
            return (2*m-1)*x*Pn(m-1, x)/m - (m-1)*Pn(m-2, x)/m
        
def Legendre(a,b,m,x):
        return np.sqrt((2*m+1)/(b-a))*Pn(m, 2*(x-b)/(b-a)+1)

def Cosine(a,b,m,x):
    square_root_term = np.sqrt(1 / (b-a) * 8 * math.pi * m / (math.sin(4 * math.pi * m) + 4 * math.pi * m))
    outer_term = np.cos(2 * math.pi * m * (x - a) / (b - a))
    return square_root_term * outer_term

## ------------------ BASIS FUNCTIONS END ------------------ ##