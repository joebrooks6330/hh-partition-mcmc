from pandas import read_csv
from numpy import array,genfromtxt,zeros,concatenate
from HH_case_partition_MCMC import RunPartitionsMCMC,IndexChange1dTo2d
from partition_functions import FlatPartition,get_simple_dataset
from pickle import dump
from os.path import isfile
from scipy.stats import norm,poisson
from scipy.optimize import minimize
import numpy as np



data_fn = "CARAZO_2021_FCDATASET.csv"
dataset_H =array(genfromtxt("datasets\\" + data_fn,delimiter=','))

k_max = len(dataset_H)-1
m = IndexChange1dTo2d(k_max)[0]
dataset_L = get_simple_dataset(dataset_H,m)

quebec_census_data = np.array([138280, 44770,41545,17555*2/3,17555/3]) #(assume 5+ person hh are split 2/3 5 person and 1/3 6 person)
quebec_census_data_sw = np.array([cd*i for i,cd in enumerate(quebec_census_data,2)])
quebec_census_data_proportions = quebec_census_data_sw/sum(quebec_census_data_sw)

alpha_0 = 200 #Concentration parameter
alpha = alpha_0*quebec_census_data_proportions

dot_for_contacts = concatenate([zeros(n+1)+n for n in range(1,m+1)])

n_iters = int(5e7)
p_beta_move= 0.01

#phi(-t) is MGF of infectious period distribution
infectious_period_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                                     "Markov": lambda t: 1/(1+t),
                                     "Gamma2": lambda t: (1+(t/2))**(-2)}

inf_period_str = "Fixed" 

print("Using infectious period assumption:", inf_period_str,end = "\n")
phi = infectious_period_assumption_dict[inf_period_str]

#Low information (eta not fixed)
results_fn_L = "outputs\\"  + data_fn.split(".")[0] + "_low_info_results_" + inf_period_str + ".pkl"

if isfile(results_fn_L):
    print("Low info MCMC has already been run for " + data_fn)
else:
    print("Running MCMC for low info for " + data_fn)
    C0_L = FlatPartition(dataset_L[2],dataset_L[1],dataset_L[0],m)
    low_info_results = RunPartitionsMCMC(dataset_H,0.1,0.1,m,1*n_iters,0.1,0.1,alpha,p_beta_move,thin=100,verbose=True,phi = phi)

    with open(results_fn_L,"wb") as f:
        dump(low_info_results,f)
        
#Medium Information
results_fn_M = "outputs\\"  + data_fn.split(".")[0] + "_medium_info_results_" + inf_period_str + ".pkl"

if isfile(results_fn_M):
    print("Medium info MCMC has already been run for " + data_fn)
else:
    print("Running MCMC for medium info for " + data_fn)
    medium_info_results = RunPartitionsMCMC(dataset_H,0.1,0.1,m,int(0.1*n_iters),0.1,0.1,p_beta_move=10*p_beta_move,thin=10,verbose=True,info_level="m",phi = phi)

    with open(results_fn_M,"wb") as f:
        dump(medium_info_results,f)

#High Information
results_fn_H = "outputs\\" + data_fn.split(".")[0] + "_high_info_results_" + inf_period_str + ".pkl"
if isfile(results_fn_H):
    print("High info MCMC has already been run for " + data_fn)
else:
    print("Running MCMC for high info for " + data_fn)
    high_info_results = RunPartitionsMCMC(dataset_H,0.1,0.1,m,int(n_iters*p_beta_move),0.1,0.1,p_beta_move=1.,thin=1,verbose=True,info_level="h",phi = phi)

    with open(results_fn_H,"wb") as f:
        dump(high_info_results,f)
        
quit()        