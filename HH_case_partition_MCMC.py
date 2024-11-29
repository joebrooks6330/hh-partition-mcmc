#%% Imports
import numpy as np



#%% Index change functions
def IndexChange2dTo1d(i,j):
    if i<j:
        raise ValueError("The value of i must be greater or equal to j.")
    if not (i>=0 and j>=0):
        raise ValueError("Both i and j must be positive.")
    if (type(i)!=int) or (type(j)!=int):
        raise TypeError("i and j must be integers.")
        
    k = 0.5*(i-1)*(i+2) + j
    return(int(k))

def IndexChange1dTo2d(k):
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
#%%
def MoveContact(C,k1,k2,infected):
    if k1<2:
        raise ValueError("Chosen household would be empty if individual was removed")
        
    n1,y1 = IndexChange1dTo2d(k1)
    
    if (y1==0) and infected:
        raise ValueError("Index k1 corresponds to no secondary cases")
    
    n2,y2 = IndexChange1dTo2d(k2)
    
    C_new = C.copy()
    C_new[k1] -= 1
    C_new[k2] -= 1
    if infected:
        k3 = IndexChange2dTo1d(n1-1, y1-1)
        k4 = IndexChange2dTo1d(n2+1, y2+1)
    else:
        k3 = IndexChange2dTo1d(n1-1, y1)
        k4 = IndexChange2dTo1d(n2+1, y2)
        
    C_new[k3] += 1
    C_new[k4] += 1
    return C_new

def SelectIndices(C,dot_for_contacts,m,u_inf):
    max_k = len(C)
    C_contacts = C*dot_for_contacts
    k1 = int(np.random.choice(np.arange(2,max_k),p = C_contacts[2:]/sum(C_contacts[2:])))
    
    n1,y1 = IndexChange1dTo2d(k1)
    infected = (y1/n1)>u_inf
    
    C_temp = C.copy()
    C_temp[k1]-=1
    C_temp_contacts = C_temp*dot_for_contacts
    max_k2 = int(0.5*(m+2)*(m-1))
    k2 = int(np.random.choice(np.arange(max_k2),p = C_temp_contacts[:max_k2]/sum(C_temp_contacts[:max_k2])))
    
    
    return k1,k2,infected