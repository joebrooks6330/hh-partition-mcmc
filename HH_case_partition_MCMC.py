#%% Imports
import numpy as np
from math import comb
from scipy.linalg import solve
from tqdm import tqdm



#%% Index change functions
def IndexChange2dTo1d(i,j):
    """
    Converts from 2D index to 1D index for a partition of contacts and cases. Inverse of IndexChange1dTo2d

    Parameters
    ----------
    i : int
        Number of secondary contacts
    j : int
        Number of secondary cases

    Returns
    -------
    k: int
       Corresponding 1D index value
    """
    if i<j:
        raise ValueError("The value of i must be greater or equal to j.")
    if not (i>=0 and j>=0):
        raise ValueError("Both i and j must be positive.")
    if (type(i)!=int) or (type(j)!=int):
        raise TypeError("i and j must be integers.")
        
    k = 0.5*(i-1)*(i+2) + j
    return(int(k))

def IndexChange1dTo2d(k):
    """
    Converts from 1D index to 2D index for a partition of contacts and cases. Inverse of IndexChange1dTo2d

    Parameters
    ----------
    k: int
       Corresponding 1D index value
   
    Returns
    -------
    i : int
        Number of secondary contacts
    j : int
        Number of secondary cases
    """
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
#%% Iteration functions - Functions that are run every iteration

def SelectIndices(C,dot_for_contacts,m,u_inf,n_moves=1):
    """
    For a given case and contacts partition, selects indices and infected status for the movement of an individual. 

    Parameters
    ----------
    C : np.ndarray length k_max
        Partition of contacts and cases
    dot_for_contacts : np.ndarray length k_max 
        np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
    m : int
        Maximum size of a household
    u_inf : float
        Random number between 0 and 1 used to determine infection status of the selected individual

    Returns
    -------
    k1 : int
        1D index of the type of household from which an individual will be removed
    k2 : int
        1D index of the type of household from which an individual will be added
    infected : bool
        Boolean value indicating if the individual being moved is infected
    proposal_prob : float
        Probability of selecting the proposed move given each individual s

    """
    max_k = len(C)
    proposal_prob = 1
    C_temp = C.copy()
    max_k2 = int(0.5*(m+2)*(m-1))
    
    remove_indices = np.zeros(n_moves)
    place_indices = np.zeros(n_moves)
    infected = np.zeros(n_moves)

    for i in range(n_moves):
        C_temp_contacts = C_temp*dot_for_contacts
        k1 = int(np.random.choice(np.arange(2,max_k),p = C_temp_contacts[2:]/sum(C_temp_contacts[2:])))
        proposal_prob *= C_temp_contacts[k1]/sum(C_temp_contacts[2:])
        C_temp[k1]-=1
        remove_indices[i] = k1
        
        k2 = int(np.random.choice(np.arange(max_k2),p = C_temp_contacts[:max_k2]/sum(C_temp_contacts[:max_k2])))
        proposal_prob *= C_temp_contacts[k2]/sum(C_temp_contacts[:max_k2])
        C_temp[k2]-=1
        place_indices[i] = k2
        
        n1,y1 = IndexChange1dTo2d(k1)
        inf_check = (y1/n1)>u_inf[i]
        if inf_check:
            proposal_prob*= y1/n1
            infected[i] = 1
        else:
            proposal_prob*= 1-(y1/n1)
    
    
   
        
    
    return remove_indices,place_indices,infected,proposal_prob
def MoveContact(C,k1,k2,infected):
    """
    For a given case and contacts partition, indices and infected status returns a new partition for moving one individual of that infected status from a household of one type to another.

    Parameters
    ----------
    C : np.ndarray length max_k
        Partition of contacts and cases
    k1 : int
        1D index of the type of household from which an individual will be removed
    k2 : int
        1D index of the type of household from which an individual will be added
    infected : bool
        Boolean value indicating if the individual being moved is infected

    Returns
    -------
    C_new : np.ndarray length max_k
        New partition following the moving of an individual

    """
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


def RevProposalProbability(C_proposed,C_current,dot_for_contacts,k1,k2,infected,m):
    """
    Calculates the probability of proposing the current partition from the newly proposed one.

    Parameters
    ----------
    C_proposed : np.ndarray length max_k
        Proposed new partition
    C_current : np.ndarray length max_k
        Current partition from previous accepted particle
    dot_for_contacts : np.ndarray length k_max 
        np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
    k1 : int
        1D index of the type of household from which an individual will be removed
    k2 : int
        1D index of the type of household from which an individual will be added
    infected : bool
        Boolean value indicating if the individual being moved is infected
    m : int
        Maximum size of a household

    Returns
    -------
    proposal_prob : TYPE
        DESCRIPTION.

    """
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
    """
    Calculates the final size distribution for a given number of initial susceptible and infected individuals. All individuals are identical and there assumed to be no new introductions after the initial cases.

    Parameters
    ----------
    n : int
        Initial number of susceptible
    m : int
        Initial Number of infected
    beta : float
        Person-to-person rate of transmission
    phi : func
        Moment generating function of the infectious period (e.g. for constant infectious period pass lamdba t: np.exp(-t))

    Returns
    -------
    P : np.ndarray
        Final size probability distribution
    """
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
    """
    Calculates the log-likelihood for a given partition and transmission parameter. Assume frequency based mixing and constant infectious period.

    Parameters
    ----------
    C : np.ndarray length k_max
        Partition of contacts and cases
    beta : float
        Person-to-person rate of transmission
    m : int
        Maximum size of a household

    Returns
    -------
    ll : TYPE
        DESCRIPTION.

    """
    ll= 0 
    
    phi = lambda t: np.exp(-t) # Constant infectious period
    fs = [final_size_distribution_homogeneous_no_intro(n, 1, beta/n, phi) for n in range(1,m+1)]
    for k,c in enumerate(C):
        n,y = IndexChange1dTo2d(k)
        ll += c*np.log(fs[n-1][y])
    return ll

#%% Run MCMC
def RunPartitionsMCMC(C0,beta,m,n_iters):
    """
    Runs an MCMC over

    Parameters
    ----------
    C : np.ndarray length k_max
        Initial partition of contacts and cases
    beta : float
        Person-to-person rate of transmission
    m : int
        Maximum size of a household
    n_iters : int
        Number of iterations the MCMC will run for

    Returns
    -------
    C : np.ndarray (n_iters+1,max_k)
        Accepted partition at each iteration including the initial partition.
    likelihoods : np.ndarray n_iters+1
        Log-likelihood of each accepted partition. Calculated by PartitionLogLikelihood

    """
    dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
    max_k = len(C0)
    
    #Generate random numbers
    u = np.random.uniform(0,1,size=n_iters) #Accept/reject proposals
    u_infected = np.random.uniform(0,1,size=n_iters) #Determine infected status of each proposed move
    
    
    C = np.zeros((n_iters+1,max_k))
    C[0] = C0
    likelihoods = np.zeros(n_iters+1)
    likelihoods[0] = PartitionLogLikelihood(C0, beta, m)

    for i in tqdm(range(n_iters),desc = "Running MCMC"):
        k1,k2,infected,proposal_prob = SelectIndices(C[i], dot_for_contacts, m, u_infected[i]) #Select move
        C_proposed = MoveContact(C[i], k1, k2, infected) #Generate new partition given the proposed move
        
        reverse_proposal_prob = RevProposalProbability(C_proposed, C[i], dot_for_contacts, k1, k2, infected, m)
        
        llh_proposed = PartitionLogLikelihood(C_proposed, beta, m)
        llhA = llh_proposed - likelihoods[i]
        proposalA = np.log(proposal_prob)-np.log(reverse_proposal_prob)
        
        #Decide accept of reject
        A = llhA+proposalA
        if A>np.log(u[i]):
            C[i+1] = C_proposed
            likelihoods[i+1] = llh_proposed
        else:
            C[i+1]= C[i]
            likelihoods[i+1] = likelihoods[i]
        
            
    return C,likelihoods