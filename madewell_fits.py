import pandas as pd
import numpy as np
from numpy import array
from os.path import isfile
from HH_case_partition_MCMC import RunPartitionsMCMC,IndexChange1dTo2d
from partition_functions import FlatPartition
from multiprocessing import Pool
from tqdm import tqdm
from argparse import ArgumentParser
from pickle import dump,load
from scipy.stats import norm,beta

filepath = 'datasets/household_studies_low_info_madewell.csv'

if isfile(filepath):
    df = pd.read_csv(filepath)
else:
    print(f"File {filepath} not found.")
    quit()

eta_fixed = 0.


data_to_fit = array(df[['Number of Households', 'Number of household contacts',
       'Number of household secondary cases',
       'Maximum number of household contacts']])
data_to_fit = array([np.insert(v,0,i) for i,v in enumerate(data_to_fit)]) #Add index for datasets


results_fn = "outputs/madewell_fits_results"
results_suffix = f"_eta={eta_fixed}"
results_fn = results_fn + results_suffix + ".pkl"

class MW_datasets_chain():
     def __init__(self):
        pass
    
     def __call__(self,p):
        
        
        N = p[2]
        n = p[3]
        y = p[4]
        m = p[5]
        beta0 = 1.0
        eta0 = eta_fixed
        #eta_logprior = beta(1.01,2).logpdf
        
        p_beta_move = 10/N
        n_iters = int(1000*N)
        try:
            C0 = FlatPartition(n,y,N,m)
            results = RunPartitionsMCMC(C0,
                                        beta0+norm(0,0.1).rvs(),
                                        eta0,
                                        int(m),
                                        n_iters,
                                        0.1,
                                        0.,
                                        p_beta_move = p_beta_move,
                                        thin = 10,
                                        verbose = False,
                                        info_level = 'l')

        except:
            import traceback
            print("Chain", p, "failed.")
            print("Error details:")
            traceback.print_exc()
            results = 0
        return results



def main(no_of_workers,n_chains):
    data_to_fit_repeated = np.repeat(data_to_fit,n_chains,axis=0)
    data_to_fit_repeated = array([np.insert(v,0,i) for i,v in enumerate(data_to_fit_repeated)]) #Add chain index
    index_arr = data_to_fit_repeated[:,:2]
    print("Starting fits (eta fixed = ", eta_fixed, ") with", no_of_workers, "workers and", n_chains, "chains per dataset. Total number of chains:", len(list(data_to_fit_repeated)))

    chain = MW_datasets_chain()

    with Pool(no_of_workers) as pool:
        results = list(tqdm(pool.imap(chain, data_to_fit_repeated), total=len(data_to_fit_repeated), desc="Running chains",smoothing=0))
    
    with open(results_fn, "wb") as f:
        dump([index_arr,results],f)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_of_workers', type=int, default=10)
    parser.add_argument('--n_chains',
                        type=int,
                        default=10)
    args = parser.parse_args()

    main(args.no_of_workers,
         args.n_chains)

