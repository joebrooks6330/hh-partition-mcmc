from pandas import read_csv
from numpy import array,genfromtxt,zeros,concatenate
from HH_case_partition_MCMC import RunPartitionsMCMC,IndexChange1dTo2d
from partition_functions import FlatPartition,get_simple_dataset
from pickle import dump
from os.path import isfile
from scipy.stats import norm
import numpy as np



data_fn = "CARAZO_2021_FCDATASET.csv"
dataset_H =array(genfromtxt("datasets\\" + data_fn,delimiter=','))


k_max = len(dataset_H)-1
m = IndexChange1dTo2d(k_max)[0]
dataset_L = get_simple_dataset(dataset_H,m)


dot_for_contacts = concatenate([zeros(n+1)+n for n in range(1,m+1)])
partition_prior = array([sum(dataset_H*(dot_for_contacts==n)) for n in range(1,m+1)])

n_iters = int(1e7)
p_beta_move=0.01

#phi(-t) is MGF of infectious period distribution
phi = lambda t: np.exp(-t) # Constant infectious period
phi = lambda t: 1/(1+t) # Markov
a=2
phi = lambda t: (1+t/a)**(-a) #General gamma(a,a)





#Low information (eta not fixed)
results_fn_L = "outputs\\" + data_fn.split(".")[0] + "_low_info_results_Gamma2.pkl"

if isfile(results_fn_L):
    print("Low info MCMC has already been run for " + data_fn)
else:
    print("Running MCMC for low info for " + data_fn)
    C0_L = FlatPartition(dataset_L[2],dataset_L[1],dataset_L[0],m)
    low_info_results = RunPartitionsMCMC(C0_L,0.01,0.01,m,1*n_iters,0.1,0.1,p_beta_move,thin=100,verbose=True,phi = phi)

    with open(results_fn_L,"wb") as f:
        dump(low_info_results,f)

#Low information (eta not fixed, priored partition)
results_fn_L_priored = "outputs\\" + data_fn.split(".")[0] + "_low_info_size_dist_priored_results.pkl"

if isfile(results_fn_L_priored):
    print("Low info MCMC has already been run for " + data_fn)
else:
    print("Running MCMC for low info for " + data_fn)
    C0_L = FlatPartition(dataset_L[2],dataset_L[1],dataset_L[0],m)
    low_info_results = RunPartitionsMCMC(C0_L,0.01,0.01,m,n_iters,0.1,0.1,p_beta_move,thin=100,verbose=True,partition_prior=partition_prior,phi = phi)

    with open(results_fn_L_priored,"wb") as f:
        dump(low_info_results,f)

#Low information (eta priored)
for eta in [0,0.5,0.7,1]:
    results_fn_L_eta = "outputs\\" + data_fn.split(".")[0] + "_low_info_results_eta=" + str(eta) +"_prior.pkl"
    if isfile(results_fn_L_eta):
        print("Low info MCMC has already been run for " + data_fn + " with fixed eta = " + str(eta))
    else:
        print("Running MCMC for low info for " + data_fn + " with fixed eta = " + str(eta))
        C0_L = FlatPartition(dataset_L[2],dataset_L[1],dataset_L[0],m)
        low_info_results = RunPartitionsMCMC(C0_L,0.1,0.1,m,n_iters,0.1,0.1,p_beta_move,thin=100,verbose=True,eta_logprior=norm(eta,0.1).logpdf,phi = phi)# type: ignore 

        with open(results_fn_L_eta,"wb") as f:
            dump(low_info_results,f)


#Medium Information
results_fn_M = "outputs\\" + data_fn.split(".")[0] + "_medium_info_results_Gamma2.pkl"

if isfile(results_fn_M):
    print("Medium info MCMC has already been run for " + data_fn)
else:
    print("Running MCMC for medium info for " + data_fn)
    medium_info_results = RunPartitionsMCMC(dataset_H,0.1,0.1,m,n_iters,0.1,0.1,p_beta_move,thin=100,verbose=True,info_level="m",phi = phi)

    with open(results_fn_M,"wb") as f:
        dump(medium_info_results,f)

#High Information
results_fn_H = "outputs\\" + data_fn.split(".")[0] + "_high_info_results_Gamma2.pkl"

if isfile(results_fn_H):
    print("High info MCMC has already been run for " + data_fn)
else:
    print("Running MCMC for high info for " + data_fn)
    high_info_results = RunPartitionsMCMC(dataset_H,1,1,m,int(10*n_iters*p_beta_move),0.1,0.1,1.,thin=1,verbose=True,info_level="h",phi = phi)

    with open(results_fn_H,"wb") as f:
        dump(high_info_results,f)