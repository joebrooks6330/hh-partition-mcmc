#%% Imports
import numpy as np
from math import comb
from scipy.linalg import solve
from tqdm import tqdm



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
#%% Iteration functions
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
    proposal_prob = 1
    C_contacts = C*dot_for_contacts
    k1 = int(np.random.choice(np.arange(2,max_k),p = C_contacts[2:]/sum(C_contacts[2:])))
    proposal_prob *= C_contacts[k1]/sum(C_contacts[2:])
    n1,y1 = IndexChange1dTo2d(k1)
    infected = (y1/n1)>u_inf
    if infected:
        proposal_prob*= y1/n1
    else:
        proposal_prob*= 1-(y1/n1)
    
    C_temp = C.copy()
    C_temp[k1]-=1
    C_temp_contacts = C_temp*dot_for_contacts
    max_k2 = int(0.5*(m+2)*(m-1))
    k2 = int(np.random.choice(np.arange(max_k2),p = C_temp_contacts[:max_k2]/sum(C_temp_contacts[:max_k2])))
    proposal_prob *= C_temp_contacts[k2]/sum(C_temp_contacts[:max_k2])
    
    
    return k1,k2,infected,proposal_prob

def ProposalProbability(C_proposed,C_current,dot_for_contacts,k1,k2,infected,m):
    C_proposed_contacts = C_proposed*(dot_for_contacts)
    proposal_prob = 1
    proposal_prob *=  C_proposed_contacts[k2]/sum(C_proposed_contacts[2:])
    C_temp = C_proposed.copy()
    C_temp[k2]-=1
    C_temp_contacts = C_temp*dot_for_contacts
    max_k1 = int(0.5*(m+2)*(m-1))
    proposal_prob *= C_temp_contacts[k1]/sum(C_temp_contacts[:max_k1])
    n1,y1 = IndexChange1dTo2d(k2)
    if infected:
        proposal_prob*= y1/n1
    else:
        proposal_prob*= 1-(y1/n1)
    return proposal_prob
    
#%%% Likelihood Functions
def final_size_distribution_homogeneous_no_intro(n,m,beta,phi):
    B = np.zeros((n+1,n+1))
    for j in range(n+1):
        for w in range(j+1):
            B[j,w] = comb(j,w)/(comb(n,w)*phi((n-j)*beta)**(m+w))
            if B[j,w] == np.inf:
                print(j,w)
    ones = np.ones(n+1)        
    P = solve(B, ones, lower= True)

    return(P)

def PartitionLogLikelihood(C,beta,m):
    ll= 0 
    
    phi = lambda t: np.exp(-t)
    fs = [final_size_distribution_homogeneous_no_intro(n, 1, beta/n, phi) for n in range(1,m+1)]
    for k,c in enumerate(C):
        n,y = IndexChange1dTo2d(k)
        ll += c*np.log(fs[n-1][y])
    return ll

#%% Run MCMC
def RunPartitionsMCMC(C0,beta,m,n_iters):
    dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
    max_k = len(C0)
    u = np.random.uniform(0,1,size=n_iters)
    u_infected = np.random.uniform(0,1,size=n_iters)
    C = np.zeros((n_iters+1,max_k))
    C[0] = C0
    likelihoods = np.zeros(n_iters+1)
    likelihoods[0] = PartitionLogLikelihood(C0, beta, m)

    for i in tqdm(range(n_iters),desc = "Running MCMC"):
        k1,k2,infected,proposal_prob = SelectIndices(C[i], dot_for_contacts, m, u_infected[i])
        C_proposed = MoveContact(C[i], k1, k2, infected)
        
        reverse_proposal_prob = ProposalProbability(C_proposed, C[i], dot_for_contacts, k1, k2, infected, m)
        
        llh_proposed = PartitionLogLikelihood(C_proposed, beta, m)
        llhA = llh_proposed - likelihoods[i]
        proposalA = np.log(proposal_prob)-np.log(reverse_proposal_prob)
        
        A = llhA+proposalA
        if A>np.log(u[i]):
            C[i+1] = C_proposed
            likelihoods[i+1] = llh_proposed
        else:
            C[i+1]= C[i]
            likelihoods[i+1] = likelihoods[i]
        
            
    return C,likelihoods