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

beta_values = [0.2,0.5,1.5]
eta_values = [0,0.5,1] 
N_hh = 100
detail_values = ["l"]#"m","h"]#,"m","h"]
I_dist_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                          "Markov": lambda t: 1/(1+t),
                          "Gamma2": lambda t: (1+(t/2))**(-2)}
I_Dist_str = "Gamma2"

 #
beta0 = 1.0 
N_datasets = 100

p_beta_move = {25:  {0.2: 0.5, 0.5: 0.5, 1.5: 0.5},
	           100: {0.2: 0.5, 0.5: 0.5, 1.5: 0.1},
	           1000:{0.2:0.2, 0.5:0.2, 1.5: 0.1}}[N_hh]
thin = {25: int(5), 100: int(25), 1000: int(100)}[N_hh] #50 for Nhh=100 10 for Nhh = 25, 100 for Nhh=1000
beta_samples = {25:100000, 100:500000, 1000:1000000}[N_hh] #100000 for Nhh=25, 500000 for Nhh=100 and 2000000 for Nhh=1000




hh_dist_str = "UK"  # "UK" or "split"
C_example = synthetic_datasets[I_Dist_str][beta_values[0]][eta_values[0]][N_hh][hh_dist_str][0]
m = IndexChange1dTo2d(len(C_example)-1)[0]


alpha_0 = 100
alpha_shape = "false" #jeffreys, true or false





params_fn = f"outputs/synth_Nhh={N_hh}/synthetic_data_validation_{hh_dist_str}_alpha={alpha_shape}_I_dist={I_Dist_str}_Nhh={N_hh}_params"
results_fn = f"outputs/synth_Nhh={N_hh}/synthetic_data_validation_{hh_dist_str}_alpha={alpha_shape}_I_dist={I_Dist_str}_Nhh={N_hh}_results"


if isfile("datasets/household_size_distributions.pkl"):
    with open("datasets/household_size_distributions.pkl", "rb") as f:
        household_size_distributions = load(f)
    if alpha_shape=="jeffreys":
        alpha = 0.5*np.ones(len(household_size_distributions[hh_dist_str]))
    if alpha_shape  == "true":
        alpha = np.array([i*p for i,p in enumerate(household_size_distributions[hh_dist_str],2)])
        alpha = alpha_0*alpha/sum(alpha)
    if alpha_shape == "false":
        false_alpha_str = "UK" if hh_dist_str == "split" else "split"
        alpha = np.array([i*p for i,p in enumerate(household_size_distributions[false_alpha_str],2)])
        alpha = alpha_0*alpha/sum(alpha)
        
else:
    print("Household size distributions file not found, cannot run with prior")
    quit()


params = list(product(beta_values,eta_values,detail_values))



class syntethic_detail_comparison_chain():
     def __init__(self,p1,seeds):
        self.beta = p1[0]
        self.eta = p1[1]
        self.detail_level = p1[2]
        self.seeds = seeds
    
     def __call__(self,p2):
        dataset_i = p2[0]
        chain_i = p2[1]
        seed = self.seeds[dataset_i,chain_i]
        np.random.seed(seed)
        
        try:
            done = False
            scale = 2 if self.beta==1.5 else 5
            # while not done:
                 
            n_iters = int(beta_samples/p_beta_move[self.beta])
            C_data = synthetic_datasets[I_Dist_str][self.beta][self.eta][N_hh][hh_dist_str][dataset_i]
            m = IndexChange1dTo2d(len(C_data)-1)[0]
            test_beta = RunPartitionsMCMC(C_data,
                                          5,
                                          self.eta,
                                          m,
                                          int(n_iters/5),
                                          np.array([[0.1,0],[0,0]]),
                                          alpha, 
                                          p_beta_move[self.beta],
                                          thin=1,
                                          verbose = False, 
                                          info_level= self.detail_level,
                                          phi = I_dist_assumption_dict[I_Dist_str])[2]
            v = np.var(test_beta[int(len(test_beta)/2):])
            SIGMA = np.array([[v,0],[0,0]])*scale*(2.38**2)
            
            
            if self.detail_level == "l":
                N,y,n = get_simple_dataset(C_data,m)
                C0 = FlatPartition(n,y,N,m)
            else:
                C0 = C_data


            results = RunPartitionsMCMC(C0,
                                        norm(5,0.1).rvs(),
                                        self.eta,
                                        m,
                                        n_iters,
                                        SIGMA,
                                        alpha = alpha,
                                        p_beta_move = p_beta_move[self.beta],
                                        thin = int(thin/p_beta_move[self.beta]),
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
    print("Household distribution:", hh_dist_str, " \nalpha:", alpha, " \nI distribution:", I_Dist_str, " \nN_hh:", N_hh)
    print()
    dataset_chain_indices = list(product(range(N_datasets),range(n_chains)))
    

    for p1 in params:
        beta = p1[0]
        eta = p1[1]
        detail = p1[2]
        seeds = np.random.randint(0,int(1e8),size = (N_datasets,n_chains))

        filename = results_fn + f"_beta={beta}_eta={eta}_detail={detail}.pkl"

        if not isfile(filename):
            print(f"Running parameter set: beta = {beta} eta = {eta} detail = {detail}")
            chain = syntethic_detail_comparison_chain(p1,seeds)
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
