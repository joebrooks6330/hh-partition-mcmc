#%% Imports
import numpy as np
from scipy.optimize import brentq
from itertools import product
from HH_case_partition_MCMC import fs_distn_single_type,IndexChange2dTo1d
from os import mkdir
from os.path import isdir, isfile
from pickle import dump,load
from tqdm import tqdm

if not isdir("datasets"):
    mkdir("datasets")


#%% Houshold size distributions (UK and SPLIT)

if isfile("datasets/household_size_distributions.pkl"):
    with open("datasets/household_size_distributions.pkl","rb") as f:
        hh_size_dist_dict = load(f)
else:
    print("Household size distributions not found, generating new distributions.")
    #Source: https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/datasets/familiesandhouseholdsfamiliesandhouseholds (table 5)
    hh_dist_UK = [9698, 4526, 3961, 1249, 544]  # 2023 UK household size distribution from size 2 to 6+

    hh_dist_UK_weighted = np.array([i*p for i,p in enumerate(hh_dist_UK,2)]) #Adjust distribution so that it is weighted by the size of the household
    hh_dist_UK_weighted = hh_dist_UK_weighted/sum(hh_dist_UK_weighted)
    UK_mu = np.dot(hh_dist_UK_weighted,np.arange(2,7,1))

    hh_dist_split_weighted_func = lambda x: np.array([p*i for i,p in  enumerate([x,0.01,0.01,0.01,1-x],2)])
    to_min = lambda x: np.dot(hh_dist_split_weighted_func(x)/sum(hh_dist_split_weighted_func(x)),np.arange(2,7,1)) - UK_mu

    a = brentq(to_min,0,1,xtol = 1e-10) # Find the value of a that gives the same mean as the UK distribution when weighted by size
    hh_dist_split = [a,0.01,0.01,0.01,1-a] # type: ignore # Split household size distribution with same mean as UK distribution when weighted by size
    hh_dist_split_weighted = hh_dist_split_weighted_func(a) # Adjust distribution so that it is weighted by the size of the household

    hh_size_dist_dict = {"UK": hh_dist_UK,
                        "split": hh_dist_split}
    with open("datasets/household_size_distributions.pkl","wb") as f:
        dump(hh_size_dist_dict,f)

#%% Synthetic data generation
def GenerateSyntheticData(beta,eta,N,hh_size_dist,phi):
    """    
    Generates a synthetic dataset for a set of parameters using Ball's equations

    Parameters
    ----------
    N: int
        Number of households
    beta: float
        Parameter for transmission rate in household with one secondary contacts. transmission rate = beta/(number of contacts)**eta
    hh_size_dist: nd.array
        Distributuion of household sizes unadjusted for size bias.
    eta: float
        Mixing parameter. See beta
    Returns
    -------
    C: nd.array:
        Final size dataset. Index indicates the numhber of total secondary contacts and cases in a household.
    """
    m = len(hh_size_dist) #max hh size (not including index case)
    max_k = int(0.5*m*(m+3)) 
    
    hh_size_dist_adjusted = np.array([i*p for i,p in enumerate(hh_size_dist,2)]) #Adjust distribution so that it is weighted by the size of the household
    hh_size_dist_adjusted = hh_size_dist_adjusted/sum(hh_size_dist_adjusted) # type: ignore
    
    household_sizes = np.random.choice(np.arange(1,m+1,1),p = hh_size_dist_adjusted,size=N) #Pick household sizes from distribution (not including index case)
    household_size_counts = [sum(np.where(household_sizes == n,1,0)) for n in range(1,m+1)]
    
    C = np.zeros(max_k)
    for n in range(1,m+1):
        fs_P = fs_distn_single_type(n, 1, beta/n**eta, phi)
        final_sizes = np.random.choice(np.arange(0,n+1,1),p=fs_P,size= household_size_counts[n-1])
        final_size_counts = [sum(np.where(final_sizes == k,1,0)) for k in range(0,n+1)]
        for k in range(0,n+1):
            index = IndexChange2dTo1d(n, k)
            C[index] += final_size_counts[k]
    return C

if isfile("datasets/synthetic_100.pkl"):
    print("Synthetic datasets have already been generated.")
else: 
    print("Synthetic datasets not found, generating new datasets.")

    beta_values = [round(x,2) for x in np.arange(0.1,2.6,0.1)]
    eta_values = [round(x,2) for x  in np.arange(0,1.25,0.25)] #[0,0.2,0.5,0.7,1.0]
    N_hh_values = [25,100,250,1000]
    hh_size_distributions_keys = ["UK","split"]
    I_dist_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                                     "Markov": lambda t: 1/(1+t),
                                     "Gamma2": lambda t: (1+(t/2))**(-2)}
    N_datasets = 100



    datasets = {I_dist:{beta: {eta: {N: {hh_dist_k:[]
                                    for hh_dist_k in hh_size_distributions_keys}
                                for N in N_hh_values}
                        for eta in eta_values}
                    for beta in beta_values}
                for I_dist in I_dist_assumption_dict.keys()}
    
    for theta in tqdm(product(beta_values,eta_values,N_hh_values,hh_size_distributions_keys,I_dist_assumption_dict.keys())):
        for i in range(N_datasets):
            data = GenerateSyntheticData(theta[0],theta[1],theta[2],hh_size_dist_dict[theta[3]],I_dist_assumption_dict[theta[4]])
            datasets[theta[4]][theta[0]][theta[1]][theta[2]][theta[3]].append(data)

    with open("datasets/synthetic_100.pkl","wb") as f:
        dump(datasets,f)


