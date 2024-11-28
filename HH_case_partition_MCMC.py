#%% Imports
import numpy as np

#%% Index change functions
def indexChange2dTo1d(i,j):
    if i<j:
        raise ValueError("The value of i must be greater or equal to j.")
    if not (i>=0 and j>=0):
        raise ValueError("Both i and j must be positive.")
    if (type(i)!=int) or (type(j)!=int):
        raise TypeError("i and j must be integers.")
        
    k = 0.5*(i-1)*(i+2) + j
    return(int(k))

def indexChange1dTo2d(k):
    if k<0:
        raise ValueError("k must be positive.")
    if type(k)!=int:
        raise TypeError("k must be an integer.")
        
    if k<2:
        i = 1
    elif k<5:
        i = 2
    else:
        i = np.ceil((-3+np.sqrt(9+8+8*k))/2)
        
    return(int(i),int(k-0.5*(i-1)*(i+2)))
    


