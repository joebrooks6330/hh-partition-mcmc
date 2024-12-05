#%% Imports
from HH_case_partition_MCMC import *
import matplotlib.pyplot as plt
import numpy as np
import winsound
from time import sleep
from scipy.stats import gaussian_kde


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

def SplitPartition(n,y,N,m):
    C = np.zeros(int(0.5*m*(m+3)))
    max_k =len(C)
    
    N_m = int(np.floor((n-N)/(m-1)))
    N_1 = int(N-np.ceil((n-N)/(m-1)))
    last_size = N-N_m-N_1
    
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
            y-= N_m*m
            C[IndexChange2dTo1d(last_size, y)] = 1
            
            
        return C
        

def PlotContactsCasesByHHSize(C,m,ax):
    contacts = np.zeros(m)
    cases = np.zeros(m)
    for n in range(1,m+1):
        contacts[n-1] = n*sum(C[np.where(dot_for_contacts==n)])
        cases[n-1] = (C[np.where(dot_for_contacts==n)]).dot(np.arange(n+1))
        SAR = cases[n-1]/contacts[n-1]
        plt.text(n+0.5, contacts[n-1]+10,"SAR = " + str(round(SAR,2)))
    print(contacts)
    ax.bar(np.arange(2,m+2),contacts,label = "Contacts")
    ax.bar(np.arange(2,m+2),cases,label = "Cases")
    ax.set_xlabel("Household size")
    ax.set_ylabel("Count")
    ax.legend()

def FinishedBeep(n_beeps,duration,freq):
    for i in range(n_beeps):
        winsound.Beep(freq, duration)
        sleep(duration/2000)
        
#%% Sandbox

N=100
beta = 0.5
hh_size_dist = [0.25,0.45,0.15,0.1,0.05]
m = len(hh_size_dist)


dot_for_cases = np.concatenate([np.arange(0,n+1) for n in range(1,m+1)])
dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])





n_iters = 100000
n_tests = 20

fig_comb,ax_comb = plt.subplots()
max_comb = -1
ax_comb.set_title("N = " + str(N) + ", " + r"$\beta=$" + str(beta) )
ax_comb.vlines([beta],-1,100,linestyle = "--",label = "Actual value")
ax_comb.set_ylabel("Density")
ax_comb.set_xlabel("Transmission Parameter" + r"$(\beta )$")

for i in range(n_tests):
    print("Test " + str(i+1))
    C_true = GenerateSyntheticData(N, beta, hh_size_dist)
    n = C_true.dot(dot_for_contacts)
    y = C_true.dot(dot_for_cases)
    C0 = FlatPartition(n, y, N, m)
    llh_true = PartitionLogLikelihood(C_true, beta, m)
    C_results,llhs,betas = RunPartitionsMCMC(C0, 1, m, n_iters,0.5)






    fig1,ax1 = plt.subplots()
    ax1.plot(llhs)
    ax1.hlines([llh_true],0,n_iters,linestyle = "--",label = "True partition LL",color = "blue")
    ax1.set_xlim(0,n_iters)
    ax1.set_ylabel("LL")
    plt.savefig("Tests/N=100, beta=0.5/llh" + str(i) + ".png")
    plt.close(fig1)
    
    beta_samples = np.random.choice(betas[20000:],size=10000)
    KDE = gaussian_kde(beta_samples)
    X = np.linspace(beta-1,beta+1,1000)
    Y = KDE(X)
    
    fig2,ax2 = plt.subplots()
    ax2.plot(X,Y,label = "KDE",color = "black")
    ax2.set_title("N = " + str(N) + ", " + r"$\beta=$" + str(beta) )
    ax2.set_ylim(0,max(Y)*1.2)
    ax2.vlines([beta],-1,100,linestyle = "--",label = "Actual value")
    ax2.set_ylabel("Density")
    ax2.set_xlabel("Transmission Parameter" + r"$(\beta )$")
    plt.savefig("Tests/N=100, beta=0.5/KDE" + str(i) + ".png")
    plt.close(fig2)
    
    fig3,ax3 = plt.subplots()
    ax3.plot(betas)
    ax3.set_xlim(0,n_iters)
    ax3.set_title(r"$\beta$" + " trace plot" )
    ax3.hlines([beta],-1,n_iters*1.2,linestyle = "--",label = "Actual value")
    ax3.set_ylabel("Accepted Particles")
    plt.savefig("Tests/N=100, beta=0.5/trace" + str(i) + ".png")
    plt.close(fig3)
    
    max_comb = max(max_comb,max(Y))
    ax_comb.plot(X,Y,color = "black", alpha = 0.2)
    ax_comb.set_ylim(0,max_comb*1.2)
    
    
        
    FinishedBeep(i+1,250,600)
fig_comb.savefig("Tests/N=100, beta=0.5/KDE_comb.png")
print("Done")
FinishedBeep(1,2000,600)