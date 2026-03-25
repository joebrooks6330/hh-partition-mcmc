from pickle import load,dump
from os.path import isfile
from itertools import product
from multiprocessing import Pool
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.stats import norm
from HH_case_partition_MCMC import RunPartitionsMCMC, IndexChange1dTo2d
from partition_functions import *


if isfile("datasets/synthetic_100.pkl"):
    with open("datasets/synthetic_100.pkl", "rb") as f:
        synthetic_datasets = load(f)
else: 
    print("Synthetic datasets haven't been generated")
    quit()

beta_values = [0.2,0.5,1.5,2.]
eta_values = [0,0.5,1] 
N_hh = 1000
detail_values = ["l"]#,"m","h"]
I_dist_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                          "Markov": lambda t: 1/(1+t),
                          "Gamma2": lambda t: (1+(t/2))**(-2)}
I_Dist_str = "Markov"

n_iters = int(1e6)
beta0 = 1.0 
p_beta_move = 0.01
N_datasets = 100

hh_dist_str = "UK"  # "UK" or "split"
C_example = synthetic_datasets[I_Dist_str][beta_values[0]][eta_values[0]][N_hh][hh_dist_str][0]
m = IndexChange1dTo2d(len(C_example)-1)[0]
alpha_0 = 100 #Scaling variable for Dirichlet parameters. 100 or 1000


params_fn = f"outputs/synthetic_data_validation_{hh_dist_str}_alpha_0={alpha_0}_I_dist={I_Dist_str}_params"
results_fn = f"outputs/synthetic_data_validation_{hh_dist_str}_alpha_0={alpha_0}_I_dist={I_Dist_str}_results"


if isfile("datasets/household_size_distributions.pkl"):
    with open("datasets/household_size_distributions.pkl", "rb") as f:
        household_size_distributions = load(f)
    alpha = np.array([i*p for i,p in enumerate(household_size_distributions[hh_dist_str],2)])
    alpha = alpha_0*alpha/sum(alpha)
else:
    print("Household size distributions file not found, cannot run with prior")
    quit()


params = list(product(beta_values,eta_values,detail_values))



class syntethic_detail_comparison_chain():
     def __init__(self,p1):
        self.beta = p1[0]
        self.eta = p1[1]
        self.detail_level = p1[2]
    
     def __call__(self,p2):
        N_hh = 1000
        dataset_i = p2[0]
        chain_i = p2[1]
        
        try:
            C_data = synthetic_datasets[I_Dist_str][self.beta][self.eta][N_hh][hh_dist_str][dataset_i]
            m = IndexChange1dTo2d(len(C_data)-1)[0]
            if self.detail_level == "l":
                N,y,n = get_simple_dataset(C_data,m)
                C0 = FlatPartition(n,y,N,m)
            else:
                C0 = C_data


            results = RunPartitionsMCMC(C0,
                                        beta0+norm(0,0.1).rvs(),
                                        self.eta,
                                        m,
                                        n_iters,
                                        0.1,
                                        0,
                                        alpha = alpha,
                                        p_beta_move = p_beta_move,
                                        thin = 100,
                                        verbose = False,
                                        info_level = self.detail_level,
                                        phi = I_dist_assumption_dict[I_Dist_str])


        except:
            import traceback
            print("Chain", p2, "failed.")
            print("Error details:")
            traceback.print_exc()
            results = 0
        
        
        
        
        return results



def main(no_of_workers,n_chains):
    print("Starting parameter sweep with", no_of_workers, "workers and", n_chains, "chains per each of the ", N_datasets, " synthetic datasets. Total number of parameter sets:", len(list(params)))
    print("Household distribution:", hh_dist_str, " alpha:", alpha, " I distribution:", I_Dist_str)
    dataset_chain_indices = list(product(range(N_datasets),range(n_chains)))


    for p1 in params:
        chain = syntethic_detail_comparison_chain(p1)
        beta = p1[0]
        eta = p1[1]
        detail = p1[2]

        filename = results_fn + f"_beta={beta}_eta={eta}_detail={detail}.pkl"

        if not isfile(filename):
            print(f"Running parameter set: beta = {beta} eta = {eta} detail = {detail}")

            with Pool(no_of_workers) as pool:
                results = list(tqdm(pool.imap(chain, dataset_chain_indices), total=len(dataset_chain_indices), desc="Running chains",smoothing=0))
            
            with open(filename, "wb") as f:
                dump(results,f)
        else:
            print(f"Results for parameter set: beta = {beta} eta = {eta} detail = {detail} already exist, skipping")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=10)
    parser.add_argument('--n_chains',
                        type=int,
                        default=10)
    args = parser.parse_args()

    main(args.no_of_workers,
         args.n_chains)
