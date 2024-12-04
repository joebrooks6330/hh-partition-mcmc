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

def FlatPartition(n,y,N,m):
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
    k_more_s = IndexChange2dTo1d(hh_size_s, cases_more)
    k_less_l = IndexChange2dTo1d(hh_size_l, cases_less)
    k_more_l = IndexChange2dTo1d(hh_size_l, cases_more)
    if N_less<=N_s:
        C[k_less_s] = N_less
        if N_more<= N_s-N_less:
            C[k_more_s] = N_more
        else:
            C[k_more_s] = N_s-N_less
            C[k_more_l] = N_more - C[k_more_s]
    else:
        C[k_less_s] = N_s
        C[k_less_l] = N_less- N_s
        C[k_more_l] = N_more
    return C

def PlotContactsCasesByHHSize(C,ax):
    contacts = np.zeros(m)
    cases = np.zeros(m)
    for n in range(1,m+1):
        contacts[n-1] = n*sum(C[np.where(dot_for_contacts==n)])
        cases[n-1] = (C[np.where(dot_for_contacts==n)]).dot(np.arange(n+1))
        SAR = cases[n-1]/contacts[n-1]
        plt.text(n+0.5, contacts[n-1]+10,"SAR = " + str(round(SAR,2)))
    ax.bar(np.arange(2,m+2),contacts,label = "Contacts")
    ax.bar(np.arange(2,m+2),cases,label = "Cases")
    ax.set_xlabel("Household size")
    ax.set_ylabel("Count")
    ax.legend()
#%% Sandbox

N=1000
beta = 0.5
hh_size_dist = [0.5,0.25,0.15,0.05,0.05]
m = len(hh_size_dist)
C_true = GenerateSyntheticData(N, beta, hh_size_dist)

dot_for_cases = np.concatenate([np.arange(0,n+1) for n in range(1,m+1)])
dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])

n = C_true.dot(dot_for_contacts)
y = C_true.dot(dot_for_cases)

C_flat = FlatPartition(n, y, N, m)

n_iters = 100000

C_start_true,llhs_start_true = RunPartitionsMCMC(C_true, beta, m, n_iters)
C_start_flat,llhs_start_flat = RunPartitionsMCMC(C_flat, beta, m, n_iters)

fig1,ax1 = plt.subplots()
ax1.plot(llhs_start_flat,label = "Start flat")
ax1.plot(llhs_start_true,label = "Start true")
plt.plot([0,n_iters*2],[llhs_start_true[0],llhs_start_true[0]],linestyle = "--")
plt.xlim(0,n_iters)
ax1.set_ylabel("LL")
ax1.legend()

fig2,ax2 = plt.subplots()
PlotContactsCasesByHHSize(C_true, ax2)
for i,c in enumerate(C_start_true[int(n_iters*0.1)::int(n_iters/100)]):
    contacts = np.zeros(m)
    cases = np.zeros(m)
    for n in range(1,m+1):
        contacts[n-1] = n*sum(c[np.where(dot_for_contacts==n)])
        cases[n-1] = (c[np.where(dot_for_contacts==n)]).dot(np.arange(n+1))
    ax2.plot(np.arange(2,m+2),contacts,alpha=0.1,color = "green")
    
final_size_distribution_homogeneous_no_intro(1, 1, beta, lambda t: np.exp(-t))
