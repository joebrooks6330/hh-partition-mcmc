# region load libraries
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
# endregion

# region Load Madewell et al. data and set eta and phi assumptions
filepath = 'datasets/household_studies_low_info_madewell.csv'

if isfile(filepath):
    df = pd.read_csv(filepath)
else:
    print(f"File {filepath} not found.")
    quit()
    
fixed = False
eta_fixed = 0.5
alpha_str = "data" #"data", "flat" or "split"

infectious_period_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                                     "Markov": lambda t: 1/(1+t),
                                     "Gamma2": lambda t: (1+t/2)**(-2)}
inf_period_str = "Gamma2" 
phi = infectious_period_assumption_dict[inf_period_str]

alphas = df.alpha.to_list()
alphas = [np.array([float(a) for a in alpha.split(",")]) for alpha in alphas]

if alpha_str == "data":
    alphas_sw = [np.array([a*i for i,a in enumerate(alpha,1)]) for alpha in alphas]
    alphas_sw = [alpha[1:]/np.sum(alpha[1:]) for alpha in alphas_sw]
elif alpha_str == "flat":
    alphas_sw = [np.ones(len(alpha)-1)/(len(alpha)-1) for alpha in alphas]
elif alpha_str == "split":
    with open("datasets/household_size_distributions.pkl","rb") as f:
        hh_size_dist_dict = load(f)
    alphas = [np.ones(len(alpha)-1) for alpha in alphas]
    for alpha in alphas:
        alpha[0] = 90
        alpha[-1] = 15
    alphas_sw = [np.array([a*i for i,a in enumerate(alpha,1)]) for alpha in alphas]
    alphas_sw = [alpha/np.sum(alpha) for alpha in alphas_sw]

    


data_to_fit = array(df[['Number of Households', 'Number of household contacts',
       'Number of household secondary cases']])
data_to_fit = array([np.insert(v,0,i) for i,v in enumerate(data_to_fit)]) #Add index for datasets


results_fn = "outputs/madewell_fits_results"
if fixed:
    results_suffix = f"_eta={eta_fixed}_{inf_period_str}_alpha_{alpha_str}"
    eta_sigma = 0.
else:
    results_suffix = f"_Carazo_eta_{inf_period_str}_alpha_{alpha_str}"
    eta_posterior_fn = "outputs/eta_gamma_posterior_Carazo_high_info_" + inf_period_str + ".pkl"
    if isfile(eta_posterior_fn):
        with open(eta_posterior_fn,"rb") as f:
            eta_posterior = load(f)
    else:
        print("Gamma distribution not computed for eta from Carazo et al. high info fit")
        quit()


results_fn = results_fn + results_suffix + ".pkl"

# endregion

class MW_datasets_chain():
     def __init__(self):
        pass
    
     def __call__(self,p):
        
        
        N = p[2]
        n = p[3]
        y = p[4]
        alpha0 = 100
        alpha = alpha0*alphas_sw[p[1]]
        m = len(alpha)
        beta0 = 5
        eta0 = eta_fixed
        #eta_logprior = beta(1.01,2).logpdf
        
        if N<50:
            p_beta_move = 0.5
        else:
            p_beta_move = 0.2
            
        n_iters = int(max(1000*N/p_beta_move,1e5))
        try:
            if N>3000:#Carazo et al. dataset
                data_fn = "CARAZO_2021_FCDATASET.csv"
                results_fn_H = "outputs/" + data_fn.split(".")[0] + "_high_info_results_" + inf_period_str + ".pkl"
                if isfile(results_fn_H):
                    with open(results_fn_H,"rb") as f:
                        results = load(f)
            else:
                C0 = FlatPartition(n,y,N,m)
                eta_logprior = eta_posterior.logpdf 
                test_beta,test_eta = RunPartitionsMCMC(C0,
                                                    beta0,
                                                    eta0,
                                                    int(m),
                                                    int(n_iters/5),
                                                    np.array([[0.1,0],[0,0.1]]),
                                                    alpha,
                                                    p_beta_move,
                                                    thin=1,
                                                    verbose = False,
                                                    info_level="l",
                                                    eta_logprior=eta_logprior,
                                                    phi=phi)[2:4]
                Sigma = (2.38**2)*np.cov(np.array([test_beta[int(n_iters/10):],test_eta[int(n_iters/10):]]))/2
                            
                results = RunPartitionsMCMC(C0,
                                            beta0+norm(0,0.1).rvs(),
                                            eta0,
                                            int(m),
                                            n_iters,
                                            Sigma,
                                            alpha=alpha,
                                            p_beta_move = p_beta_move,
                                            thin = 10,
                                            verbose = False,
                                            info_level = 'l',
                                            phi = phi,
                                            eta_logprior = eta_posterior.logpdf if not fixed else  lambda e: 0 if (e>=0 and e<=1) else -np.inf)# type: ignore

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
    print("Starting fits with ", inf_period_str ,"infectious period assumption and", no_of_workers, "workers and", n_chains, "chains per dataset. Total number of chains:", len(list(data_to_fit_repeated)))
    if fixed:
        print(f"Using fixed eta = {eta_fixed}")
    else:
        print("Using eta from Carazo et al.")
        
    print(f"Results Filename = {results_fn}")
    
    if isfile(results_fn):
        print(f"File {results_fn} already exists.")
        quit()

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

