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
#%% Iteration functions - Functions that are run every iteration

def SelectIndices(C,dot_for_contacts,m,u_inf,info_level):
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
    if info_level not in ["low","medium","l","m"]:
        raise ValueError("Invalid value for info_level")

    C_temp = C.copy()
    C_temp_contacts = C_temp*dot_for_contacts

    log_proposal_prob = 0

    #Choose first index 
    max_k1 = len(C)
    p1 = C_temp_contacts[2:]/sum(C_temp_contacts[2:])
    k1 = int(np.random.choice(np.arange(2,max_k1),p = p1)) 
    #Can't select hoyseholds with one contact because 
    #low info: removing one leaves household empty 
    #medium info: Swapping between households with one contact doesn't change anything
    n1,y1 = IndexChange1dTo2d(k1)
    log_proposal_prob += np.log(C_temp_contacts[k1]) - np.log(sum(C_temp_contacts[2:]))
    
    
    
    
    #Choose infectious status of individual
    inf_check = (y1/n1)>u_inf
    if inf_check:
        log_proposal_prob += np.log(y1) - np.log(n1)
        infected = 1
    else:
        log_proposal_prob += np.log(n1-y1) - np.log(n1)
        infected = 0
    
    #Choose second index
    if info_level[0] == "l":
        min_k2 = 0
        max_k2 = int(0.5*(m+2)*(m-1))-1
        C_temp[k1] -= 1
    if info_level[0] == "m":
        if infected:
            min_k2 = IndexChange2dTo1d(n1,0)
            max_k2 = IndexChange2dTo1d(n1,n1-1)
        else:   
            min_k2 = IndexChange2dTo1d(n1,1)
            max_k2 = IndexChange2dTo1d(n1,n1)
    
    p2 = C_temp[min_k2:max_k2+1]/sum(C_temp[min_k2:max_k2+1]) # type: ignore
    k2 = int(np.random.choice(np.arange(min_k2,max_k2+1),p = p2)) # type: ignore
    log_proposal_prob += np.log(C_temp[k2]) - np.log(sum(C_temp[min_k2:max_k2])) # type: ignore
    
   
        
    
    return k1,k2,infected,log_proposal_prob



def MoveContact(C,k1,k2,infected,info_level):
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
    if C[k1] == 0:
        raise ValueError("Chosen household bin must have atleast one household")
    
    n2,y2 = IndexChange1dTo2d(k2)
    
    C_new = C.copy()

    if info_level[0] == "l":
        C_new[k1] -= 1
        C_new[k2] -= 1
        if infected:
            k3 = IndexChange2dTo1d(n1-1, y1-1)
            k4 = IndexChange2dTo1d(n2+1, y2+1)
        else:
            k3 = IndexChange2dTo1d(n1-1, y1)
            k4 = IndexChange2dTo1d(n2+1, y2)
            
        
    
    if info_level[0] == "m":
        if k1==k2 and C_new[k1] == 1:
            return C_new
        
        C_new[k1] -= 1
        C_new[k2] -= 1
        if infected:
            k3 = IndexChange2dTo1d(n1,y1-1)
            k4 = IndexChange2dTo1d(n2,y2+1)
        else:
            k3 = IndexChange2dTo1d(n1,y1+1)
            k4 = IndexChange2dTo1d(n2,y2-1)
    
    C_new[k3] += 1 # type: ignore
    C_new[k4] += 1 # type: ignore
    return C_new


def RevProposalProbability(C_proposed,dot_for_contacts,remove_index,place_index,infected,m,info_level):
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
    C_temp = C_proposed.copy()
    C_temp_contacts = C_temp*(dot_for_contacts)
    log_proposal_prob = 0
    
            
    n1,y1 = IndexChange1dTo2d(remove_index)
    n2,y2 = IndexChange1dTo2d(place_index)

    if info_level[0] == "l":
        rev_remove_n = n2+1
        rev_place_n = n1-1
        rev_remove_y = y2
        rev_place_y = y1
    
        if infected:
            rev_remove_y +=1
            rev_place_y -=1
        
    
    if info_level[0] == "m":
        rev_remove_n = n2
        rev_place_n = n1
        rev_remove_y = y2
        rev_place_y = y1
        if infected:
            rev_remove_y +=1
            rev_place_y -= 1
        else:
            rev_remove_y -= 1
            rev_place_y += 1
    
    rev_place_k = IndexChange2dTo1d(rev_place_n, rev_place_y) # type: ignore
    rev_remove_k = IndexChange2dTo1d(rev_remove_n, rev_remove_y) # type: ignore
    log_proposal_prob +=  np.log(C_temp_contacts[rev_remove_k]) - np.log(sum(C_temp_contacts[2:]))

    if infected:
        log_proposal_prob += np.log(rev_remove_y) - np.log(rev_remove_n) # type: ignore
    else:
        log_proposal_prob += np.log(rev_remove_n - rev_remove_y) - np.log(rev_remove_n) # type: ignore
    
    if info_level[0] == "l":
        min_k2 = 0
        max_k2 = int(0.5*(m+2)*(m-1))
        C_temp[rev_remove_k]-=1
    if info_level[0] == "m":
        if infected:
            min_k2 = IndexChange2dTo1d(rev_remove_n,0) # type: ignore
            max_k2 = IndexChange2dTo1d(rev_remove_n,rev_remove_n-1) # type: ignore
        else:   
            min_k2 = IndexChange2dTo1d(rev_remove_n,1) # type: ignore
            max_k2 = IndexChange2dTo1d(rev_remove_n,rev_remove_n) # type: ignore
    log_proposal_prob += np.log(C_temp[rev_place_k]) - np.log(sum(C_temp[min_k2:max_k2])) # type: ignore
    
    
    return log_proposal_prob
    
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

def LogFactorial(a):
    """
    Calculates log(a!)

    Parameters
    ----------
    a : int

    Returns
    -------
    result : result
        log(a!)
    """
    if not (type(a) == int and a>0):
        raise TypeError("a must be a positive integer")
    result = 0
    for i in range(1,a+1):
        result += np.log(i)
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

def PartitionLogLikelihood(C,beta,m,fs):
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
        for n in range(1,m+1):
            total_hh_size_n = 0
            for y in range(n+1):
                k = IndexChange2dTo1d(n, y)
                count = int(C[k])
                if count>0:
                    ll += count * np.log(fs[n-1][y])
                    ll -= LogFactorial(count)
                    total_hh_size_n += count
            if total_hh_size_n>0:
                ll += LogFactorial(total_hh_size_n)
            
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
    beta_prior = lambda b: 0 if (b>0 and b<100) else -np.inf,
    eta_prior = lambda e: 0 if (e>=0 and e<=1) else -np.inf,
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
    beta_prior : function, optional
        A function that returns the log prior probability of the transmission rate beta (default is lambda beta: 0)
    eta_prior : function, optional
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
    
    #Initialise array to store likelihoods
    likelihoods = np.zeros(n_iters+1)
    final_size_distributions = FinalSizeDistributions(m,beta0,eta0)
    final_size_distributions_proposed = final_size_distributions
    likelihoods[0] = PartitionLogLikelihood(C0, beta0, m,final_size_distributions) 

    #Initialise array to store prior probabilities for partitions
    part_prior_probs = np.zeros(n_iters+1)
    part_prior_probs[0] = PartitionPriorProbability(C[0],partition_prior,dot_for_contacts,m)

    #Initialise array to store prior probabilities for beta (transmission rate)
    beta_prior_probs = np.zeros(n_iters+1)
    beta_prior_probs[0] = beta_prior(beta0) if beta_prior is not None else 0
    
    #Initialise array to store beta values
    betas = np.zeros(n_iters+1)
    betas[0] = beta0

    #Initialise array to store prior probabilities for beta (transmission rate)
    eta_prior_probs = np.zeros(n_iters+1)
    eta_prior_probs[0] = eta_prior(eta0) if beta_prior is not None else 0
    
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
        
        if info_level[0] == "h":
            C_proposed = C[i].copy()
            k1 = 0
            k2 = 0
            infected = 0
            log_proposal_prob = 0
            log_reverse_proposal_prob = 0
        else:
            #Select indices and infected status for the proposed move to new partition
            k1,k2,infected,log_proposal_prob = SelectIndices(C[i], dot_for_contacts, m, u_infected[i],info_level) 
            C_proposed = C[i].copy() #Copy the current partition
            C_proposed = MoveContact(C_proposed, int(k1), int(k2), infected,info_level) #Generate new partition given the proposed move
            #Calculate reverse proposal probability
            log_reverse_proposal_prob = RevProposalProbability(C_proposed, dot_for_contacts, k1,k2, infected, m,info_level)
        
        #If the current iteration is the last of a set of n_moves, propose new beta and generate new final size distributions
        if i%n_moves == n_moves-1:
            beta_proposed = betas[i]+beta_proposal_offsets[i//n_moves]
            eta_proposed = etas[i]+eta_proposal_offsets[i//n_moves]
            final_size_distributions_proposed = FinalSizeDistributions(m,beta_proposed,eta_proposed)
            llh_proposed = PartitionLogLikelihood(C_proposed, beta_proposed, m,final_size_distributions_proposed) 
            beta_prior_proposed = beta_prior(beta_proposed)
            eta_prior_proposed = eta_prior(eta_proposed) 
        else:
            beta_proposed = betas[i]
            eta_proposed = etas[i]
            llh_proposed = PartitionLogLikelihood(C_proposed, beta_proposed, m,final_size_distributions)
            beta_prior_proposed = beta_prior_probs[i]
            eta_prior_proposed = eta_prior_probs[i]
        

        llhA = llh_proposed - likelihoods[i]

        proposalA = log_reverse_proposal_prob-log_proposal_prob

        part_prior_proposed = PartitionPriorProbability(C_proposed,partition_prior,dot_for_contacts,m)

        priorA = part_prior_proposed-part_prior_probs[i] + beta_prior_proposed - beta_prior_probs[i] + eta_prior_proposed-eta_prior_probs[i]


        #Decide accept of reject
       
        A = llhA + proposalA + priorA

        if A>u[i] and beta_proposed>0:
            C[i+1] = C_proposed
            likelihoods[i+1] = llh_proposed
            part_prior_probs[i+1] = part_prior_proposed
            beta_prior_probs[i+1] = beta_prior_proposed
            betas[i+1] = beta_proposed
            eta_prior_probs[i+1] = eta_prior_proposed
            etas[i+1] = eta_proposed
            final_size_distributions = final_size_distributions_proposed
            entropies[i+1] = PartitionEntropy(C_proposed, dot_for_contacts)
            
        else:
            C[i+1]= C[i]
            likelihoods[i+1] = likelihoods[i]
            part_prior_probs[i+1] = part_prior_probs[i]
            beta_prior_probs[i+1] = beta_prior_probs[i]
            betas[i+1] = betas[i]
            eta_prior_probs[i+1] = eta_prior_probs[i]
            etas[i+1] = etas[i]
            entropies[i+1] = entropies[i]

        chosen_indices[i] = [k1,k2]
            
            
    return C[::thin],likelihoods[::thin],betas[::thin],etas[::thin],part_prior_probs[::thin],beta_prior_probs[::thin],eta_prior_probs[::thin],entropies[::thin],chosen_indices[::thin]




