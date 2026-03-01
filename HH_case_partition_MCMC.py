#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy.linalg import solve
from numpy.linalg import cond
from tqdm import tqdm
from scipy.stats import norm, multinomial
from scipy.special import gammaln
from typing import Callable

np.seterr(all='raise')


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



#%% Functions for selecting indices of contacts to be moved
def SelectIndicesLowInfo(C,m,u_inf,LU_1D_to_2D):
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

    """
    
    C_temp = C.copy()
    
    #Choose first index 
    max_k1 = len(C)
    p1 = C_temp[2:]/sum(C_temp[2:])
    k1 = int(np.random.choice(np.arange(2,max_k1),p = p1)) 
    log_proposal_prob = np.log(C_temp[k1]) - np.log(sum(C_temp[2:]))
    C_temp[k1] -= 1
    
    n1,y1 = LU_1D_to_2D[k1]
    
    #Choose infectious status of individual
    inf_check = (y1/n1)>u_inf
    
    if inf_check:
        infected = 1
        log_proposal_prob += np.log(y1) - np.log(n1)
        
    else:
        infected = 0
        log_proposal_prob += np.log(n1-y1) - np.log(n1)
    
        
    
    #Choose second index
    min_k2 = 0
    max_k2 = int(0.5*(m+2)*(m-1))  -1
    
    p2 = C_temp[min_k2:max_k2+1]/sum(C_temp[min_k2:max_k2+1]) # type: ignore
    k2 = int(np.random.choice(np.arange(min_k2,max_k2+1),p = p2)) # type: ignore
    log_proposal_prob += np.log(C_temp[k2]) - np.log(sum(C_temp[min_k2:max_k2+1])) # type: ignore
    
    return k1,k2,infected,log_proposal_prob

def SelectIndicesMediumInfo(C,dot_for_cases,LU_1D_to_2D,LU_2D_to_1D):
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
        1D index of the type of household from which will swap susceptible for infected
    k2 : int
        1D index of the type of household from which will swap infected for susceptible
    infected : bool
        Boolean value indicating if the individual being moved is infected

    """
    
    C_temp = C.copy()
    C_temp_cases = C_temp*dot_for_cases

    log_proposal_prob = 0
    
    #Choose first index, somebody infected
    max_k1 = len(C)
    p1 = C_temp_cases[2:]/sum(C_temp_cases[2:]) #Choosing household with one contact doesn't change anything
    k1 = int(np.random.choice(np.arange(2,max_k1),p = p1)) 
    n1,y1 = LU_1D_to_2D[k1] 
    log_proposal_prob += np.log(C_temp_cases[k1]) - np.log(sum(C_temp_cases[2:]))
    C_temp[k1] -= 1
    C_temp_noncases = C_temp*(n1-dot_for_cases) #Update contacts after removing one individual
 
    min_k2 = int(LU_2D_to_1D[n1,0]) # type: ignore
    max_k2 = int(LU_2D_to_1D[n1,n1-1]) # type: ignore 


    if sum(C_temp[min_k2:max_k2+1]) == 0: #If there are no households of the type we want to swap with, we can't select a second index
        k2 = -1
        return k1,k2,True,0
    else:
        p2 = (C_temp_noncases[min_k2:max_k2+1])/(sum(C_temp_noncases[min_k2:max_k2+1])) # type: ignore
        k2 = int(np.random.choice(np.arange(min_k2,max_k2+1),p =p2)) 
        log_proposal_prob += np.log(C_temp_noncases[k2]) - np.log(sum(C_temp_noncases[min_k2:max_k2+1])) # type: ignore

    return k1,k2,True,log_proposal_prob



#%% Functions for moving contacts once selected
def MoveContactLowInfo(C,k1,k2,infected,LU_1D_to_2D,LU_2D_to_1D):
    """
    For a given case and contacts partition, indices and infected status returns a new partition for moving one individual of that infected status from a household of one type to another.

    Parameters
    ----------
    C : np.ndarray length max_k+1
        Partition of contacts and cases
    k1 : int
        1D index of the type of household from which an individual will be removed
    k2 : int
        1D index of the type of household from which an individual will be added
    infected : bool
        Boolean value indicating if the individual being moved is infected

    Returns
    -------
    C_new : np.ndarray length max_k+1
        New partition following the moving of an individual

    """
    C_new = C.copy()
    C_new[k1] -= 1
    C_new[k2] -= 1
    n1,y1 = LU_1D_to_2D[k1]
    n2,y2 = LU_1D_to_2D[k2]
    if infected:
        k3 = int(LU_2D_to_1D[n1-1, y1-1])
        k4 = int(LU_2D_to_1D[n2+1, y2+1])
    else:
        k3 = int(LU_2D_to_1D[n1-1, y1])
        k4 = int(LU_2D_to_1D[n2+1, y2])

    C_new[k3] +=  1 # type: ignore
    C_new[k4] += 1 # type: ignore
    
    return C_new

def MoveContactMediumInfo(C,k1,k2):
    """
    For a given case and contacts partition, indices and infected status returns a new partition for moving one individual of that infected status from a household of one type to another.

    Parameters
    ----------
    C : np.ndarray length max_k+1
        Partition of contacts and cases
    k1 : int
        1D index of the type of household from which an individual will be removed
    k2 : int
        1D index of the type of household from which an individual will be added
    infected : bool
        Boolean value indicating if the individual being moved is infected

    Returns
    -------
    C_new : np.ndarray length max_k+1
        New partition following the moving of an individual

    """
    
    C_new = C.copy()

    if k2 == -1:
        return C_new
    else:      
        C_new[k1] -= 1
        C_new[k2] -= 1
        C_new[k1-1] += 1
        C_new[k2+1] += 1
    return C_new

#%% Function for calculating the reverse proposal probability 
def ReverseProposalProbabilityLowInfo(C_proposed,k1,k2,infected,m,LU_1D_to_2D,LU_2D_to_1D):
    """
    Calculates the probability of proposing the current partition from the newly proposed one.

    Parameters
    ----------
    C_proposed : np.ndarray length max_k+1
        Proposed new partition
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
    proposal_prob : float
        Logarithm of the reverse proposal probability

    """
    C_temp = C_proposed.copy()
    
    n1,y1 = LU_1D_to_2D[k1]
    n2,y2 = LU_1D_to_2D[k2]

    log_rev_proposal_prob = 0

    if infected:
        k3 = int(LU_2D_to_1D[n1-1, y1-1])
        k4 = int(LU_2D_to_1D[n2+1, y2+1])
        log_rev_proposal_prob += np.log(y2+1) - np.log(n2+1)


    else:
        k3 = int(LU_2D_to_1D[n1-1, y1])
        k4 = int(LU_2D_to_1D[n2+1, y2])
        log_rev_proposal_prob += np.log(n2+1-y2) - np.log(n2+1)

    
    log_rev_proposal_prob += np.log(C_temp[k4]) - np.log(sum(C_temp[2:]))
    C_temp[k4] -= 1
    min_k2 = 0
    max_k2 = int(0.5*(m+2)*(m-1))  -1

    log_rev_proposal_prob += np.log(C_temp[k3]) - np.log(sum(C_temp[min_k2:max_k2+1]))


    return log_rev_proposal_prob

def ReverseProposalProbabilityMediumInfo(C_proposed,dot_for_cases,k1,k2,LU_1D_to_2D,LU_2D_to_1D):
    """
    Calculates the probability of proposing the current partition from the newly proposed one.

    Parameters
    ----------
    C_proposed : np.ndarray length max_k+1
        Proposed new partition
    dot_for_cases : np.ndarray length max_k+1 
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
    proposal_prob : float
        Logarithm of the reverse proposal probability

    """
    if k2 == -1:
        return 0
    else:
        k3 = k1 - 1 
        k4 = k2 + 1 
        n1,y1 = LU_1D_to_2D[k1]
        max_k = int(LU_2D_to_1D[n1,n1-1])
        min_k = int(LU_2D_to_1D[n1,0])


        C_proposed_cases = C_proposed*dot_for_cases

        log_rev_proposal_prob = np.log(C_proposed_cases[k4]) - np.log(sum(C_proposed_cases[2:]))
        C_temp = C_proposed.copy()
        C_temp[k4] -= 1
        C_temp_noncases = C_temp*(n1-dot_for_cases) #Update contacts after removing one individual
        log_rev_proposal_prob += np.log(C_temp_noncases[k3]) - np.log(sum(C_temp_noncases[min_k:max_k+1])) # type: ignore

        return log_rev_proposal_prob


    
#%%% Likelihood Functions
def fs_distn_single_type(n: int,
                         m: int,
                         beta: float,
                         phi: Callable,
                         p_esc: float = 1):
    """Calculates the final size distribution for an epidemic with a single type of individual. 
       Implements the formula from Addy et al. 1991 equation (4) in single type special case.

    Args:
        n (int): The number of individuals who are intially susceptible.
        m (int): The number of individuals who are initially infected.
        beta (float): The person-to-person transmission rate parameter.
        phi (callable): The Laplace transform of the infectious period distribution. 
        p_esc (float, optional): The probability of escaping external infection from outside. Defaults to 1 i.e. self contained outbreak.
    Returns:
        P (array): An array of length n+1 where P[j] is the probability that j of the n initially susceptible individuals are ultimately infected.
    """
    B = np.zeros((n+1,n+1))
    for j in range(n+1):
        for w in range(j+1):
            B[j,w] = comb(j,w)/(comb(n,w)*phi((n-j)*beta)**(m+w)*p_esc**(n-j))
            if B[j,w] == np.inf:
                print(j,w)
           
    if cond(B)> 1e10:  
        #Matrix is ill-conditioned due to high transmission rate, approximate soluation with certain full final size
        P = np.zeros(n+1)
        P[-1] = 1
    else:    
        ones = np.ones(n+1) 
        P = solve(B, ones, lower= True)

    return(P)

def LogFactorials(N):
    """
    Calculates log(a!) for 

    Parameters
    ----------
    a : int

    Returns
    -------
    result : result
        log(a!)
    """

    if not (type(N) == int and N>0):
        raise TypeError("N must be a positive integer")
    result = np.zeros(N+1)
    for i in range(1,N+1):
        result[i] = result[i-1] + np.log(i)
    return result

def LogGammas(N, alpha):
    m = len(alpha)
    result = np.zeros((m,N+1))
    result[:,0] = np.zeros(m) + gammaln(alpha-1)
    for n in range(m):
        for i in range(1,N+1):
            if alpha[n]-2+i > 0:
                result[n,i] = result[n,i-1] + np.log(alpha[n]-2+i)
            else:
                result[n,i] = gammaln(alpha[n]-1+i)
    return result


def LogFinalSizeDistributions(m: int,
                           beta: float,
                           eta: float =1.0,
                           phi = lambda t: np.exp(-t)):
    """
    Returns a list of final size distributitons for each of the household sizes from 1 to m. Assumes frequency based mixing and constant infectious period.
    
    Parameters
    ----------
    m : int 
        Maximum size of a household
    beta : 
        float Person-to-person rate of transmission
    Returns
    -------
    fs : list
        List of final size distributions for each household size from 1 to m
    """
    fs = np.concatenate([fs_distn_single_type(n, 1, beta/(n**eta), phi) for n in range(1,m+1)])
    log_fs = np.array([np.log(fs_k) if fs_k>0 else -1e10 for fs_k in fs])
    return log_fs

def PartitionLogLikelihood(C,beta,m,log_fs,LF,LG,dot_for_contacts):
    """
    Calculates the log-likelihood for a given partition and transmission parameter.

    Parameters
    ----------
    C : np.ndarray length k_max
        Partition of contacts and cases
    beta : float
        Person-to-person rate of transmission
    m : int
        Maximum size of a household
    fs : list
        List of final size distributions for each household size from 1 to m

    Returns
    -------
    ll : float
        Log-likelihood of the partition given the parameter and 

    """
    if beta<0:
        return -np.inf
    else:
        log_fs_c = log_fs.copy()

        ll = C.dot(log_fs_c)
        
        for k in range(len(C)):
                count = int(C[k])
                if count>0:
                    ll -= LF[count]
        for n in range(m):
            filter = dot_for_contacts==(n+1)
            N_n = int(sum(C*filter))
            ll+= LG[n,N_n]
        if ll==np.inf:
            print()
            print(log_fs_c)
            print()
            print(LG)
            print()
            print(LF)
            print()
            
        return ll
        



#%% Run MCMC
def RunPartitionsMCMC(C0: np.ndarray,
    beta0: float,
    eta0: float,
    m: int,
    n_iters: int,
    beta_proposal_sd: float,
    eta_proposal_sd: float,
    alpha: np.ndarray= np.zeros(1),
    p_beta_move: float = 0.1,
    thin: int = 1,
    verbose: bool = True,
    beta_logprior = lambda b: 0 if (b>0 and b<10) else -np.inf,
    eta_logprior = lambda e: 0 if (e>=0 and e<=1) else -np.inf,
    info_level: str = "l",
    phi = lambda t: np.exp(-t)
    ):
    """
    Given a number of primary cases, secondary contacts and cases (encoded in C0), this function runs an MCMC to fit a transmission rate.
    Aswell as running a traditional MetHast MCMC with Gaussian proposals, it also runs through partitions of the secondary cases/contacts
    into households, assuming 1 primary case per household, that are compatible with the data.

    Parameters
    ----------
    C0 : np.ndarray length k_max (k_max = 0.5*(m+2)*(m-1))
        Initial partition of contacts and cases
    beta0 : float
        Initial parameter value for person-to-person rate of transmission rate
    eta0 : float
        Intitial parameter value that can move between frequency (eta=1) and density dependent (eta=0) mixing.
    m : int
        Maximum size of a household
    n_iters : int
        Number of iterations the MCMC will run for
    beta_proposal_sd: float
        Standard deviation of the Gaussian proposal distribution for the transmission parameter
    p_beta_move : int
        The probability of an MCMC step proposing a new beta as oppose to a move for the partition
    verbose: bool
        If True, tqdm loading bar is shown for MCMC 
    hh_size_dist : np.ndarray
        A probability vector giving the prior distribution of household sizes from 1 to m. If size_weighted is False, this is converted to a size-weighted distribution internally. 
        First entry is the number of households with 2 indiviudals (1 primary case 1 contact)
    size_weighted : bool
        If True, hh_size_dist is treated as a size-weighted distribution already. If False it is converted to a size-weighted distribution internally.
    beta_logprior : function, optional
        A function that returns the log prior probability of the transmission rate beta (default is lambda beta: 0)
    eta_logprior : function, optional
        A function that returns the log prior probability of the eta parameter (default is lambda eta: 0)
    
    Returns
    -------
    C : np.ndarray (n_iters+1,max_k+1)
        Accepted partition at each iteration including the initial partition.
    likelihoods : np.ndarray n_iters+1
        Log-likelihood of each accepted partition. Calculated by PartitionLogLikelihood

    """

    #Type checking
    if not isinstance(C0, np.ndarray):
        raise ValueError("C0 must be a numpy array")
    if not isinstance(beta0, (int, float)) or beta0 <= 0:
        raise ValueError("beta0 must be a positive number")
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not isinstance(n_iters, int) or n_iters <= 0:
        raise ValueError("n_iters must be a positive integer")
    if not isinstance(beta_proposal_sd, (int, float)) or beta_proposal_sd < 0:
        raise ValueError("beta_proposal_sd must be a positive number")
    if not isinstance(p_beta_move, float) or p_beta_move < 0 or p_beta_move>1 :
        raise ValueError("p_beta_move must be a probability")
    if info_level not in ["low","medium","high","l","m","h"]:
        raise ValueError("Invalid value for info_level")


    if (alpha == np.zeros(1)).all():
        #Default alpha for no prior information on household size distribution is a flat Dirichlet prior
        alpha = np.zeros(m)+100
    elif len(alpha)!=m:
        raise ValueError("alpha must have length m")
    
    dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
    dot_for_cases = np.concatenate([np.arange(0, n + 1) for n in range(1, m + 1)])

    max_k = len(C0)-1
    max_k_check = IndexChange2dTo1d(m,m)
    if max_k != max_k_check:
        raise ValueError("Length of C0 is not correct for the given value of m. Length of C0 must be " + str(max_k_check))
    
    #Generate random numbers
    u = np.log(np.random.uniform(0,1,size=n_iters)) #Accept/reject proposals
    u_move_type = np.random.uniform(0,1,size=(n_iters))
    move_type_beta_bools = u_move_type<p_beta_move
    u_infected = np.random.uniform(0,1,size=(n_iters)) #Determine infected status of each proposed move
    
    #Generate random offsets for beta proposals all at once
    beta_proposal_offsets = norm(0,beta_proposal_sd).rvs(size=sum(move_type_beta_bools))
    eta_proposal_offsets = norm(0,eta_proposal_sd).rvs(size=sum(move_type_beta_bools))
    i_offsets = 0
    
    #Initialise arrays to store partitions
    C = np.zeros((n_iters+1,max_k+1),dtype=int)
    C[0] = C0
    
    #Precalculate log factorials for likelihood calculations
    LG = LogGammas(int(sum(C0)),alpha) 
    LF = LogFactorials(int(sum(C0))) #Log factorials for likelihood calculations

    #Precalculate look-up arrays for moving between 1D and 2D indexing
    k_max = int(0.5*(m+3)*(m))
    LU_1D_to_2D = np.array([IndexChange1dTo2d(int(k)) for k in range(k_max)])
    LU_2D_to_1D = np.zeros((m+1,m+1))-0.5
    for n in range(1,m+1):
        for y in range(n+1):
            LU_2D_to_1D[n,y] = IndexChange2dTo1d(n,y)
        
    #Initialise array to store likelihoods
    likelihoods = np.zeros(n_iters+1,dtype = np.float32)

    log_final_size_distributions = LogFinalSizeDistributions(m,beta0,eta0,phi)

    log_final_size_distributions_proposed = log_final_size_distributions
    likelihoods[0] = PartitionLogLikelihood(C0, beta0, m,log_final_size_distributions,LF,LG,dot_for_contacts)

    #Initialise array to store prior probabilities for beta (transmission rate)
    beta_logprior_probs = np.zeros(n_iters+1,dtype = np.float32)
    beta_logprior_probs[0] = beta_logprior(beta0) if beta_logprior is not None else 0
    
    #Initialise array to store beta values
    betas = np.zeros(n_iters+1,dtype = np.float32)
    betas[0] = beta0

    #Initialise array to store prior probabilities for beta (transmission rate)
    eta_logprior_probs = np.zeros(n_iters+1,dtype = np.float32)
    eta_logprior_probs[0] = eta_logprior(eta0) if beta_logprior is not None else 0
    
    #Initialise array to store beta values
    etas = np.zeros(n_iters+1,dtype = np.float32)
    etas[0] = eta0


    #Initialise array for CHOSEN INDICESAdd commentMore actions
    #chosen_indices = np.zeros((n_iters,2))



    #Start loop, displaying a loading bar if verbose is True
    for i in (tqdm(range(n_iters),desc = "Running MCMC",mininterval=5) if verbose else range(n_iters)):
        #Select indices and infected status for the proposed move to new partition
        C_proposed = C[i].copy() #Copy the current partition     
        #If the current iteration is the last of a set of n_moves, propose new beta and generate new final size distributions
        if move_type_beta_bools[i]:           
            beta_proposed = betas[i]+beta_proposal_offsets[i_offsets]
            eta_proposed = etas[i]+eta_proposal_offsets[i_offsets]
            i_offsets += 1
            
            log_final_size_distributions_proposed = LogFinalSizeDistributions(m,beta_proposed,eta_proposed,phi=phi)
            beta_logprior_proposed = beta_logprior(beta_proposed)
            eta_logprior_proposed = eta_logprior(eta_proposed) 
            log_reverse_proposal_prob = 0
            log_proposal_prob = 0
            k1 = 0
            k2 = 0 
        else:
            beta_proposed = betas[i]
            eta_proposed = etas[i]
            beta_logprior_proposed = beta_logprior_probs[i]
            eta_logprior_proposed = eta_logprior_probs[i]
            if info_level[0] == "l":
                k1,k2,infected,log_proposal_prob = SelectIndicesLowInfo(C[i],m, u_infected[i],LU_1D_to_2D) 
                C_proposed = MoveContactLowInfo(C_proposed, int(k1), int(k2), infected, LU_1D_to_2D,LU_2D_to_1D) #Generate new partition given the proposed move
                #Calculate reverse proposal probability
                log_reverse_proposal_prob = ReverseProposalProbabilityLowInfo(C_proposed, k1,k2, infected, m, LU_1D_to_2D,LU_2D_to_1D)
            
            elif info_level[0] == "m":
                k1,k2,infected,log_proposal_prob = SelectIndicesMediumInfo(C[i],dot_for_cases,LU_1D_to_2D,LU_2D_to_1D) 
                C_proposed = MoveContactMediumInfo(C_proposed, int(k1), int(k2)) #Generate new partition given the proposed move
                #Calculate reverse proposal probability
                log_reverse_proposal_prob = ReverseProposalProbabilityMediumInfo(C_proposed,dot_for_cases, k1,k2,LU_1D_to_2D,LU_2D_to_1D)

            elif info_level[0] == "h":
                k1 = 0
                k2 = 0 
                log_proposal_prob = 0
                log_reverse_proposal_prob = 0
            else:
                raise ValueError("Invalid value for info_level")
                
                
                
            
        llh_proposed = PartitionLogLikelihood(C_proposed, beta_proposed, m,log_final_size_distributions_proposed,LF,LG,dot_for_contacts) 

        
        
        try:
            llhA = llh_proposed - likelihoods[i]
        except:
            print(llh_proposed, likelihoods[i])

        proposalA = log_reverse_proposal_prob-log_proposal_prob

        priorA =  beta_logprior_proposed - beta_logprior_probs[i] + eta_logprior_proposed-eta_logprior_probs[i]


        #Decide accept of reject
       
        A = llhA + proposalA + priorA

        if A>u[i] and beta_proposed>0:
            C[i+1] = C_proposed
            likelihoods[i+1] = llh_proposed
            beta_logprior_probs[i+1] = beta_logprior_proposed
            betas[i+1] = beta_proposed
            eta_logprior_probs[i+1] = eta_logprior_proposed
            etas[i+1] = eta_proposed
            log_final_size_distributions = log_final_size_distributions_proposed
            
        else:
            C[i+1]= C[i]
            likelihoods[i+1] = likelihoods[i]
            beta_logprior_probs[i+1] = beta_logprior_probs[i]
            betas[i+1] = betas[i]
            eta_logprior_probs[i+1] = eta_logprior_probs[i]
            etas[i+1] = etas[i]

            
    C_results = C[::thin]   
    return C_results,likelihoods[::thin],betas[::thin],etas[::thin],beta_logprior_probs[::thin],eta_logprior_probs[::thin]

