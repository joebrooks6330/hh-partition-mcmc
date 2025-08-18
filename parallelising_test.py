from argparse import ArgumentParser
from HH_case_partition_MCMC import *
from scipy.stats import norm
from pickle import load, dump
from os.path import isdir, isfile
from numpy import array
from multiprocessing import Pool

if isfile("datasets/synthetic.pkl"):
    with open("datasets/synthetic.pkl", "rb") as f:
        synthetic_datasets = load(f)
else: 
    print("Synthetic datasets haven't been generated")
    quit()


beta = 1.
eta = 0.5
N_hh = 100
hh_dist_key = "UK"


dataset = synthetic_datasets[beta][eta][N_hh][hh_dist_key][1]





beta0 = 1
eta0 = eta
n_moves = 10
n_iters  = int(1e6)
info_level = "l"
sd = 0.1

class HH_data_MCMC_chain():
    def __init__(self, 
     
                 C0: np.ndarray,
                 beta0: float,
                 eta0: float):
        self.C0 = C0
        self.beta0 = beta0
        self.eta0 = eta0

        self.m = IndexChange1dTo2d(len(C0)-1)[0]


        if not isinstance(C0, np.ndarray):
            raise ValueError("C0 must be a numpy array")
        if not isinstance(beta0, (int, float)) or beta0 <= 0:
            raise ValueError("beta0 must be a positive number") 
        if not isinstance(eta0, (int, float)) or eta0 < 0:
            raise ValueError("eta0 must be a non-negative number")  
    
    def __call__(self,
                 p): 
        try:
            print("Now attempting chain ", p)
            this_beta0 = self.beta0 + norm(0,0.1).rvs()
            results = RunPartitionsMCMC(self.C0,
                                        this_beta0,
                                        self.eta0,
                                        self.m,
                                        n_iters,
                                        sd,
                                        0,
                                        n_moves=n_moves,
                                        verbose=False,
                                        thin = 10)
        except:
            results = 0
        
        return results

def main(no_of_workers,
         n_chains):
    chain = HH_data_MCMC_chain(dataset,beta0,eta0)

    params = range(n_chains)

    with Pool(no_of_workers) as pool:
        results = pool.map(chain, params)

    #C_codes = array([r[0] for r in results])
    #likelihoods = array([r[1] for r in results])
    #betas = array([r[2] for r in results])
    #etas = array([r[3] for r in results])

    with open("outputs/parallelising_test.pkl", "wb") as f:
        dump(results,f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=8)
    parser.add_argument('--n_chains',
                        type=int,
                        default=8)
    args = parser.parse_args()

    main(args.no_of_workers,
         args.n_chains)