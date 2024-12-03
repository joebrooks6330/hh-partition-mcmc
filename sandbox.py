#%% Imports
from HH_case_partition_MCMC import *
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")



#%% Functions
def GenerateSyntheticData(N,beta,hh_size_dist):
    m = len(hh_size_dist) #max hh size
    max_k = int(0.5*m*(m+3))
    
    hh_size_dist_adjusted = np.array([i*p for i,p in enumerate(hh_size_dist,1)])
    hh_size_dist_adjusted = hh_size_dist_adjusted/sum(hh_size_dist_adjusted)
    
    household_sizes = np.random.choice(np.arange(1,m+1,1),p = hh_size_dist_adjusted,size=N)
    household_size_counts = [sum(np.where(household_sizes == n,1,0)) for n in range(1,m+1)]
    C = np.zeros(max_k)
    for n in range(1,m+1):
        fs_P = final_size_distribution_homogeneous_no_intro(n, 1, beta/n, lambda t: np.exp(-t))
        final_sizes = np.random.choice(np.arange(0,n+1,1),p=fs_P,size= household_size_counts[n-1])
        final_size_counts = [sum(np.where(final_sizes == k,1,0)) for k in range(0,n+1)]
        for k in range(0,n+1):
            index = IndexChange2dTo1d(n, k)
            C[index] += final_size_counts[k]
    return C

def PlotContactsCasesByHHSize(C):
    contacts = np.zeros(m)
    cases = np.zeros(m)
    for n in range(1,m+1):
        contacts[n-1] = n*sum(C[np.where(dot_for_contacts==n)])
        cases[n-1] = (C[np.where(dot_for_contacts==n)]).dot(np.arange(n+1))
        SAR = cases[n-1]/contacts[n-1]
        plt.text(n+0.5, contacts[n-1]+10,"SAR = " + str(round(SAR,2)))
    plt.bar(np.arange(2,m+2),contacts,label = "Contacts")
    plt.bar(np.arange(2,m+2),cases,label = "Cases")
    plt.xlabel("Household size")
    plt.ylabel("Count")
    plt.legend()
#%% Sandbox

N=1000
beta = 0.5
hh_size_dist = [0.2,0.4,0.25,0.1,0.05]
C_true = GenerateSyntheticData(N, beta, hh_size_dist)
PlotContactsCasesByHHSize(C_true)

dot_for_cases = np.concatenate([np.arange(0,n+1) for n in range(1,m+1)])
dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])

C0 = np.arange(max_k,0,-1)
total_contacts = C0.dot(dot_for_contacts)

n_iters = 10000

C,llhs = RunPartitionsMCMC(C0, 1.5, m, n_iters)
