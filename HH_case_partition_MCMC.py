#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy.linalg import solve
from tqdm import tqdm
from scipy.stats import norm, multinomial




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

#%% Partition ID conversion functions
ascii_ords = np.concatenate([np.arange(48,58),np.arange(65,91),np.arange(97,123)])
def convert_to_ascii_ords(i):
    if i<10:
        return 48+i
    elif 10<=i<36:
        return 65+i-10
    elif 36<=i<=61:
        return 97+i-36
    else:
        return 62**3

def partition_to_ID(C):
    ID  = ""
    for c_fl in C:
        c = int(c_fl)
        if c<= 61:
            ID = ID +  "00" + chr(convert_to_ascii_ords(c))
        elif c<= 62*62 - 1:
            ID += "0" + chr(convert_to_ascii_ords(c//62)) + chr(convert_to_ascii_ords(c%62))
        elif c<=62**3 - 1:
            ID += chr(convert_to_ascii_ords(c//(62*62))) + chr(convert_to_ascii_ords((c//62)%62)) + chr(convert_to_ascii_ords(c%62))
        else:
            raise ValueError("Partition ID is too long. Maximum length is 3 characters per element.")
    return ID

convert_from_ascii_ords = {ord(c):i for i,c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")}

def ID_to_partition(ID):
    C = np.zeros(len(ID)//3)
    for i in range(len(ID)//3):
        c = ID[i*3:i*3+3]
        C[i] = convert_from_ascii_ords[ord(c[0])]*62**2+ convert_from_ascii_ords[ord(c[1])]*62 +convert_from_ascii_ords[ord(c[2])]
    return C


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
    C_proposed : np.ndarray length max_k
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
    C_proposed : np.ndarray length max_k
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

reverse_proposal_probability_dict = {"l": ReverseProposalProbabilityLowInfo,
                                     "m": ReverseProposalProbabilityMediumInfo,
                                     "h": lambda C_proposed,dot_for_cases,k1,k2,infected,m: 0}


    
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

def FinalSizeDistributions(m: int,
                           beta: float,
                           eta: float =1.0):
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
    phi = lambda t: np.exp(-t) # Constant infectious period
    fs = [final_size_distribution_homogeneous_no_intro(n, 1, beta/(n**eta), phi) for n in range(1,m+1)]
    return fs

def PartitionLogLikelihood(C,beta,m,fs,LF):
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
        ll= 0 
        k=0
        for n in range(1,m+1):
            #total_hh_size_n = 0
            for y in range(n+1):
                count = int(C[k])
                if count>0:
                    ll += count * np.log(fs[n-1][y])
                    ll -= LF[count]
                k+=1

            
        return ll
    
def PartitionEntropy(C,dot_for_contacts):
    contacts = C*(dot_for_contacts)
    total_contacts = C.dot(dot_for_contacts)
    proportions = contacts/total_contacts
    log_proportions = [np.log(c) if c!=0 else 0 for c in proportions ]
    
    return -proportions.dot(log_proportions)

def PartitionPriorProbability(C,partition_prior,dot_for_contacts,m):
    """    Calculates the prior probability of a given partition.

    Parameters
    ----------
    C : np.ndarray
        Partition of contacts and cases.
    partition_prior : np.ndarray
        Prior probabilities for each partition.
    dot_for_contacts : np.ndarray
        Array representing the dot product for contacts.
    m : int
        Maximum size of a household.

    Returns
    -------
    prior_prob : float
        Log of the prior probability of the partition.
    """
    if partition_prior is None:
        return 1
    x = np.array([C.dot(dot_for_contacts==n) for n in range(1,m+1)])
    prior_prob = multinomial.logpmf(x = x, n = sum(x), p = partition_prior)

    return prior_prob



    
#%% Plotting
def PlotPartition(C,m,dot_for_contacts,llh=None,i=None,ax=None):
    if ax == None:
        fig,ax = plt.subplots()

    contacts = np.zeros(m)
    cases = np.zeros(m)
    for n in range(1,m+1):
        contacts[n-1] = n*sum(C[np.where(dot_for_contacts==n)])
        cases[n-1] = (C[np.where(dot_for_contacts==n)]).dot(np.arange(n+1))
        SAR = cases[n-1]/contacts[n-1]
        plt.text(n+0.5, contacts[n-1]+10,"SAR = " + str(round(SAR,2)))
    if llh!=None:
        if i!=None:
            plt.text(0.5,max(contacts),str(i) + "   " + str(llh))
        else:
            plt.text(0.5,max(contacts),str(llh))
    else:
        if i!=None:
            plt.text(0.5,max(contacts),str(llh))


    ax.bar(np.arange(2,m+2),contacts,label = "Contacts")
    ax.bar(np.arange(2,m+2),cases,label = "Cases")
    ax.set_xlabel("Household size")
    ax.set_ylabel("Count")
    ax.legend()
#%% Run MCMC
def RunPartitionsMCMC(C0: np.ndarray,
    beta0: float,
    eta0: float,
    m: int,
    n_iters: int,
    beta_proposal_sd: float,
    eta_proposal_sd: float,
    n_moves: int = 1,
    thin: int = 1,
    verbose: bool = True,
    partition_prior: np.ndarray = np.zeros(1),
    beta_logprior = lambda b: 0 if (b>0 and b<100) else -np.inf,
    eta_logprior = lambda e: 0 if (e>=0) else -np.inf,
    info_level: str = "l"
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
    n_moves : int
        The number of contacts which are moved for each proposed new partition
    verbose: bool
        If True, tqdm loading bar is shown for MCMC 
    partition_prior : np.ndarray
        A probability vector of the multinomial prior on the size of households. 
    beta_logprior : function, optional
        A function that returns the log prior probability of the transmission rate beta (default is lambda beta: 0)
    eta_logprior : function, optional
        A function that returns the log prior probability of the eta parameter (default is lambda eta: 0)
    
    Returns
    -------
    C : np.ndarray (n_iters+1,max_k)
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
    if not isinstance(n_moves, int) or n_moves <= 0:
        raise ValueError("n_moves must be a positive integer")
    if info_level not in ["low","medium","high","l","m","h"]:
        raise ValueError("Invalid value for info_level")
    


    
    if (partition_prior == np.zeros(1)).all():
        #Check if partition_prior is provided, if not set to uniform prior
        partition_prior = np.ones(m)/m
    else:
        partition_prior = partition_prior/np.sum(partition_prior) #Normalise prior
    
    dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
    dot_for_cases = np.concatenate([np.arange(0, n + 1) for n in range(1, m + 1)])
    
    dot_for_dict = {"l": dot_for_contacts,
                    "m": dot_for_cases,
                    "h": dot_for_contacts}

    max_k = len(C0)
    
    #Generate random numbers
    u = np.log(np.random.uniform(0,1,size=n_iters)) #Accept/reject proposals
    u_infected = np.random.uniform(0,1,size=(n_iters)) #Determine infected status of each proposed move
    
    #Generate random offsets for beta proposals all at once
    beta_proposal_offsets = norm(0,beta_proposal_sd).rvs(size=n_iters//n_moves)
    eta_proposal_offsets = norm(0,eta_proposal_sd).rvs(size=n_iters//n_moves)
    
    #Initialise arrays to store partitions
    C = np.zeros((n_iters+1,max_k))
    C[0] = C0

    #Precalculate log factorials for likelihood calculations
    LF = LogFactorials(int(sum(C0))) #Log factorials for likelihood calculations

    #Precalculate look-up arrays for moving between 1D and 2D indexing
    k_max = int(0.5*(m+3)*(m))
    LU_1D_to_2D = np.array([IndexChange1dTo2d(int(k)) for k in range(k_max)])
    LU_2D_to_1D = np.zeros((m+1,m+1))-0.5
    for n in range(1,m+1):
        for y in range(n+1):
            LU_2D_to_1D[n,y] = IndexChange2dTo1d(n,y)
        
    #Initialise array to store likelihoods
    likelihoods = np.zeros(n_iters+1)
    final_size_distributions = FinalSizeDistributions(m,beta0,eta0)
    final_size_distributions_proposed = final_size_distributions
    likelihoods[0] = PartitionLogLikelihood(C0, beta0, m,final_size_distributions,LF)

    #Initialise array to store prior probabilities for partitions
    part_logprior_probs = np.zeros(n_iters+1)
    part_logprior_probs[0] = PartitionPriorProbability(C[0],partition_prior,dot_for_contacts,m)

    #Initialise array to store prior probabilities for beta (transmission rate)
    beta_logprior_probs = np.zeros(n_iters+1)
    beta_logprior_probs[0] = beta_logprior(beta0) if beta_logprior is not None else 0
    
    #Initialise array to store beta values
    betas = np.zeros(n_iters+1)
    betas[0] = beta0

    #Initialise array to store prior probabilities for beta (transmission rate)
    eta_logprior_probs = np.zeros(n_iters+1)
    eta_logprior_probs[0] = eta_logprior(eta0) if beta_logprior is not None else 0
    
    #Initialise array to store beta values
    etas = np.zeros(n_iters+1)
    etas[0] = eta0

    #Initialise array to store entropy values
    entropies = np.zeros(n_iters+1)
    entropies[0] = PartitionEntropy(C0, dot_for_contacts)

    #Initialise array for CHOSEN INDICESAdd commentMore actions
    chosen_indices = np.zeros((n_iters,2))



    #Start loop, displaying a loading bar if verbose is True
    for i in (tqdm(range(n_iters),desc = "Running MCMC",mininterval=5) if verbose else range(n_iters)):
        #Select indices and infected status for the proposed move to new partition
        if info_level[0] == "l":
            k1,k2,infected,log_proposal_prob = SelectIndicesLowInfo(C[i],m, u_infected[i],LU_1D_to_2D) 
            C_proposed = C[i].copy() #Copy the current partition
            C_proposed = MoveContactLowInfo(C_proposed, int(k1), int(k2), infected, LU_1D_to_2D,LU_2D_to_1D) #Generate new partition given the proposed move
            #Calculate reverse proposal probability
            log_reverse_proposal_prob = ReverseProposalProbabilityLowInfo(C_proposed, k1,k2, infected, m, LU_1D_to_2D,LU_2D_to_1D)
        
        elif info_level[0] == "m":
            k1,k2,infected,log_proposal_prob = SelectIndicesMediumInfo(C[i],dot_for_cases,LU_1D_to_2D,LU_2D_to_1D) 
            C_proposed = C[i].copy() #Copy the current partition
            C_proposed = MoveContactMediumInfo(C_proposed, int(k1), int(k2)) #Generate new partition given the proposed move
            #Calculate reverse proposal probability
            log_reverse_proposal_prob = ReverseProposalProbabilityMediumInfo(C_proposed,dot_for_cases, k1,k2,LU_1D_to_2D,LU_2D_to_1D)

        elif info_level[0] == "h":
            C_proposed = C[i].copy()
            k1 = 0
            k2 = 0
            infected = 0
            log_proposal_prob = 0
            log_reverse_proposal_prob = 0
            
       
        
        #If the current iteration is the last of a set of n_moves, propose new beta and generate new final size distributions
        if i%n_moves == n_moves-1:
            beta_proposed = betas[i]+beta_proposal_offsets[i//n_moves]
            eta_proposed = etas[i]+eta_proposal_offsets[i//n_moves]
            final_size_distributions_proposed = FinalSizeDistributions(m,beta_proposed,eta_proposed)
            llh_proposed = PartitionLogLikelihood(C_proposed, beta_proposed, m,final_size_distributions_proposed,LF) 
            beta_logprior_proposed = beta_logprior(beta_proposed)
            eta_logprior_proposed = eta_logprior(eta_proposed) 
        else:
            beta_proposed = betas[i]
            eta_proposed = etas[i]
            llh_proposed = PartitionLogLikelihood(C_proposed, beta_proposed, m,final_size_distributions,LF)
            beta_logprior_proposed = beta_logprior_probs[i]
            eta_logprior_proposed = eta_logprior_probs[i]
        

        llhA = llh_proposed - likelihoods[i]

        proposalA = log_reverse_proposal_prob-log_proposal_prob

        part_logprior_proposed = PartitionPriorProbability(C_proposed,partition_prior,dot_for_contacts,m)

        priorA = part_logprior_proposed-part_logprior_probs[i] + beta_logprior_proposed - beta_logprior_probs[i] + eta_logprior_proposed-eta_logprior_probs[i]


        #Decide accept of reject
       
        A = llhA + proposalA + priorA

        if A>u[i] and beta_proposed>0:
            C[i+1] = C_proposed
            likelihoods[i+1] = llh_proposed
            part_logprior_probs[i+1] = part_logprior_proposed
            beta_logprior_probs[i+1] = beta_logprior_proposed
            betas[i+1] = beta_proposed
            eta_logprior_probs[i+1] = eta_logprior_proposed
            etas[i+1] = eta_proposed
            final_size_distributions = final_size_distributions_proposed
            entropies[i+1] = PartitionEntropy(C_proposed, dot_for_contacts)
            
        else:
            C[i+1]= C[i]
            likelihoods[i+1] = likelihoods[i]
            part_logprior_probs[i+1] = part_logprior_probs[i]
            beta_logprior_probs[i+1] = beta_logprior_probs[i]
            betas[i+1] = betas[i]
            eta_logprior_probs[i+1] = eta_logprior_probs[i]
            etas[i+1] = etas[i]
            entropies[i+1] = entropies[i]

        chosen_indices[i] = [k1,k2]
            
    #Convert partitions to codes
    C_results = C[::thin]
    C_codes = [partition_to_ID(P) for P in C_results]       
    return C_codes,likelihoods[::thin],betas[::thin],etas[::thin],part_logprior_probs[::thin],beta_logprior_probs[::thin],eta_logprior_probs[::thin],entropies[::thin],chosen_indices[::thin]




