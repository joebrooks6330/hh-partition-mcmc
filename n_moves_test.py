from os.path import isfile
from pickle import load,dump
import numpy as np
from itertools import product
from scipy.stats import norm
from HH_case_partition_MCMC import RunPartitionsMCMC, IndexChange1dTo2d
from partition_functions import *
from multiprocessing import Pool
from argparse import ArgumentParser
from tqdm import tqdm
import sys


if isfile("datasets/synthetic.pkl"):
    with open("datasets/synthetic.pkl", "rb") as f:
        synthetic_datasets = load(f)
else: 
    print("Synthetic datasets haven't been generated")
    quit()

N_datasets = 10
n_iters = int(1e6)
beta0 = 1.0
n_moves_values = [1,10,100,1000]

params_fn = "outputs/n_moves_test_UK_params.pkl"
results_fn = "outputs/n_moves_test_UK_results.pkl"

class n_movesChain():
     def __init__(self):
        pass
    
     def __call__(self,p):
        
        
        beta = 0.5
        eta = 0.0
        N_hh = 1000
        n_moves = p[0]
        dataset_i = p[1]
        chain_i = p[2]
        
        try:
            C_data = synthetic_datasets[beta][eta][N_hh]["UK"][dataset_i]
            m = IndexChange1dTo2d(len(C_data)-1)[0]
            N,y,n = get_simple_dataset(C_data,m)
            C0 = FlatPartition(n,y,N,m)

            results = RunPartitionsMCMC(C0,
                                        beta0+norm(0,0.1).rvs(),
                                        eta,
                                        m,
                                        n_iters,
                                        0.1,
                                        0,
                                        n_moves=n_moves,
                                        verbose=False,
                                        thin = 10)
        except:
            print("Chain", p, "failed.")
            print("Error details:", sys.exc_info()[0])
            results = 0
        return results



def main(no_of_workers,n_chains):
    params = list(product(n_moves_values,range(N_datasets),range(n_chains)))
    with open(params_fn, "wb") as f:
        dump(params,f)

    print("Starting parameter sweep with", no_of_workers, "workers and", n_chains, "chains per parameter set. Total number of chains:", len(list(params)))

    chain = n_movesChain()

    with Pool(no_of_workers) as pool:
        results = list(tqdm(pool.imap(chain, params), total=len(params), desc="Running chains",smoothing=0))
    
    with open(results_fn, "wb") as f:
        dump(results,f)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=10)
    parser.add_argument('--n_chains',
                        type=int,
                        default=10)
    args = parser.parse_args()

    main(args.no_of_workers,
         args.n_chains)
