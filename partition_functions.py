import numpy as np
from HH_case_partition_MCMC import IndexChange2dTo1d


def get_simple_dataset(C,m):
    dot_for_cases = np.concatenate([np.arange(0, n + 1) for n in range(1, m + 1)])
    dot_for_contacts = np.concatenate([np.zeros(n + 1) + n for n in range(1, m + 1)])

    N = sum(C) #Number of households
    y = C.dot(dot_for_cases)
    n = C.dot(dot_for_contacts)

    return (N,y,n)

#Functions for generating flat or split partitions for starting partitions
def FlatPartition(n,y,N,m):
    if not (N>=0 and n>=0 and y>=0):
        raise ValueError("n, y and N must be non-negative integers.")
    if not (n>=y):
        raise ValueError("n must be greater or equal to y.")

    C = np.zeros(int(0.5*m*(m+3)))
    
    hh_size_s = int(n//N)
    hh_size_l = hh_size_s + 1
    
    N_l = n-hh_size_s*N
    N_s = N - N_l
    
    cases_less = int(y//N)
    cases_more = cases_less+1
    N_more = y - N*cases_less
    N_less = N - N_more

    
    k_less_s = IndexChange2dTo1d(hh_size_s, cases_less)
    if cases_more<=hh_size_s:
        k_more_s = IndexChange2dTo1d(hh_size_s, cases_more)
    else:
        k_more_s = None
    k_less_l = IndexChange2dTo1d(hh_size_l, cases_less)
    k_more_l = IndexChange2dTo1d(hh_size_l, cases_more)
    
    if N_more<=N_l:
        C[k_more_l] = N_more
        if N_less<= N_l-N_more:
            C[k_less_l] = N_less
        else:
            C[k_less_l] = N_l- N_more
            C[k_less_s] = N_less - C[k_less_l]
    else:
        C[k_more_l] = N_l
        C[k_more_s] = N_more - N_l
        C[k_less_s] = N_less

    return C

def SplitPartition(n,y,N,m):
    C = np.zeros(int(0.5*m*(m+3)))
    max_k =len(C)
    
    N_m = int(np.floor((n-N)/(m-1)))
    N_1 = int(N-np.ceil((n-N)/(m-1)))
    last_size = int(n-m*N_m-N_1)
    
    if y<N_1:
        C[0] = N_1-y
        C[1] = y
        C[IndexChange2dTo1d(last_size, 0)] = 1
        C[IndexChange2dTo1d(m, 0)] = N_m
    else:
        C[1] = N_1
        y -= N_1
        if y< m*N_m:
            C[IndexChange2dTo1d(last_size, 0)] = 1
            C[IndexChange2dTo1d(m, m)] = y//m

            if y%m ==0:
                C[IndexChange2dTo1d(m, 0)] = N_m - y//m
            else:
                C[IndexChange2dTo1d(m, int(y%m))] = 1
                C[IndexChange2dTo1d(m, 0)] =  N_m - y//m - 1
        else:
            C[IndexChange2dTo1d(m, m)] = N_m
            y-= int(N_m*m)
            C[IndexChange2dTo1d(last_size, int(y))] = 1
            
            
    return C