from pickle import load, dump
from os.path import isfile
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from itertools import product
from HH_case_partition_MCMC import IndexChange1dTo2d
from partition_functions import FlatPartition, get_simple_dataset
plt.style.use("ggplot")
plt.rcParams["font.family"] = "monospace"

#Import household size distributions used to generate synthetic data
if isfile("datasets/household_size_distributions.pkl"):
    with open("datasets/household_size_distributions.pkl","rb") as f:
        hh_size_dist_dict = load(f)
else:
    raise FileNotFoundError("Household size distributions not found, please run synthetic_data_generation.py to generate them.")

hh_dist_UK_weighted = np.array([i*p for i,p in enumerate(hh_size_dist_dict["UK"],2)]) #Adjust distribution so that it is weighted by the size of the household
hh_dist_UK_weighted = hh_dist_UK_weighted/sum(hh_dist_UK_weighted)
UK_mu = np.dot(hh_dist_UK_weighted,np.arange(2,7,1))
m = len(hh_dist_UK_weighted)

hh_dist_split_weighted = np.array([i*p for i,p in enumerate(hh_size_dist_dict["split"],2)]) #Adjust distribution so that it is weighted by the size of the household
hh_dist_split_weighted = hh_dist_split_weighted/sum(hh_dist_split_weighted)
split_mu = np.dot(hh_dist_split_weighted,np.arange(2,7,1))

fig,axs = plt.subplots(1,1,figsize = (10,5), sharey=True)
fig.suptitle("Size-weighted household size distributions", fontsize = 16)
axs.bar(np.arange(1.8,6.8,1),hh_dist_UK_weighted,width = 0.4, color = "C0",label = "UK LFS (2023)")
axs.axvline(UK_mu, color = "black", linestyle = "--",label = f"Shared mean = {round(UK_mu,2)}")
axs.set_xlabel("Household size")
axs.set_ylabel("Proportion of households")
axs.bar(np.arange(2.2,7.2,1),hh_dist_split_weighted,width = 0.4, color = "C1", label = "Split")
axs.legend()
plt.tight_layout()
plt.savefig("figures/household_size_distributions.png", dpi = 400)
plt.cla()

#import synthetic datasets
if isfile("datasets/synthetic_100.pkl"):
    with open("datasets/synthetic_100.pkl", "rb") as f:
        synthetic_datasets = load(f)


hh_dist_strings = hh_size_dist_dict.keys()
hh_dist_prior_tuples = [(hh_dist,S) for hh_dist in hh_dist_strings for S in [100,1000]]
I_dist_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                          "Markov": lambda t: 1/(1+t),
                          "Gamma2": lambda t: (1+(t/2))**(-2)}

beta_values = [0.2,0.5,1.5,2.]
eta_values = [0,0.5,1]
detail_values = ["l"]
N_datasets = 100

      
                

results_C0 = {tp: {I_dist:{beta:{eta:{detail: np.zeros((N_datasets)) for detail in detail_values} for eta in eta_values} for beta in beta_values} for I_dist in I_dist_assumption_dict.keys()} for tp in hh_dist_prior_tuples}
results_SAR = {tp: {I_dist:{beta:{eta:{detail: np.zeros(N_datasets) for detail in detail_values} for eta in eta_values} for beta in beta_values} for I_dist in I_dist_assumption_dict.keys()} for tp in hh_dist_prior_tuples}
results_beta = {tp:{I_dist: {beta:{eta:{detail: np.zeros(N_datasets) for detail in detail_values} for eta in eta_values} for beta in beta_values} for I_dist in I_dist_assumption_dict.keys()} for tp in hh_dist_prior_tuples}
results_llh = {tp:{I_dist: {beta:{eta:{detail: np.zeros(N_datasets) for detail in detail_values} for eta in eta_values} for beta in beta_values} for I_dist in I_dist_assumption_dict.keys()} for tp in hh_dist_prior_tuples}
results_beta_mean = {tp:{I_dist: {beta:{eta:{detail: np.zeros(N_datasets) for detail in detail_values} for eta in eta_values} for beta in beta_values} for I_dist in I_dist_assumption_dict.keys()} for tp in hh_dist_prior_tuples}
results_beta_CI = {tp:{I_dist: {beta:{eta:{detail: np.zeros((N_datasets,2)) for detail in detail_values} for eta in eta_values} for beta in beta_values} for I_dist in I_dist_assumption_dict.keys()} for tp in hh_dist_prior_tuples}
params = list(product(hh_dist_prior_tuples,I_dist_assumption_dict.keys(),beta_values,eta_values,detail_values))

c=0
for p1 in tqdm(params,desc = "Loading results"):
    tp = p1[0]
    hh_dist = p1[0][0]
    S = p1[0][1]
    I_dist = p1[1]
    beta = p1[2]
    eta = p1[3]
    detail = p1[4]
    filename = f"outputs/synthetic_data_validation_{hh_dist}_S={S}_I_dist={I_dist}_results"
    filename += f"_beta={beta}_eta={eta}_detail={detail}.pkl"

    if isfile(filename):
        c+=1
        datasets = synthetic_datasets[I_dist][beta][eta][1000][hh_dist]
        with open(filename, "rb") as f:
            results = load(f)
        m = IndexChange1dTo2d(len(datasets[0])-1)[0]
        dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
        dot_for_cases = np.concatenate([np.arange(0, n + 1) for n in range(1, m + 1)])

        results_C0[tp][I_dist][beta][eta][detail] = array([r[0][0] for r in results])
        simple_datasets = [get_simple_dataset(C_data,m) for C_data in datasets]
        F_datasets = [FlatPartition(n,y,N,m) for (N,y,n) in simple_datasets]
        
        
        # for i in range(10):
        #     print(f"Debugging beta={beta}, eta={eta}, detail={detail}, hh_dist={hh_dist}, prior={prior}")
        #     print(datasets[i])
        #     print(np.array(F_datasets[i]))
        #     print(np.array(results_C0[tp][beta][eta][detail][i]))
        n_iters = len(results[0][0])
        results_SAR[tp][I_dist][beta][eta][detail] = array([C.dot(dot_for_cases)/C.dot(dot_for_contacts) for C in datasets])
        results_llh[tp][I_dist][beta][eta][detail] = array([r[1] for r in results])
        results_beta[tp][I_dist][beta][eta][detail] = array([r[2][n_iters//10:] for r in results])
        results_beta_mean[tp][I_dist][beta][eta][detail] = array([np.mean(r) for r in results_beta[tp][I_dist][beta][eta][detail]]) #type:ignore
        results_beta_CI[tp][I_dist][beta][eta][detail]= array([np.percentile(r, [2.5,97.5]) for r in results_beta[tp][I_dist][beta][eta][detail]]) #type:ignore
    else:
        print(f"File {filename} not found.")

# C_results = results_C[("split",True)][1.5][0.5]["l"]
# plt.scatter(range(1,m+1),array([k*n for k,n in enumerate(hh_dist_split_weighted,2)])/sum([k*n for k,n in enumerate(hh_dist_split_weighted,2)]),s=20)
# for i in range(len(C_results)):
#     C_results_i = C_results[i]#[int(1e3):]
#     hh_size_counts = [(C_results_i*(dot_for_contacts==n)).sum(axis=1) for n in range(1,m+1)]
#     mean_counts = array([hh_size_counts[n-1].mean() for n in range(1,m+1)])/1000
#     CI_counts = array([np.percentile(hh_size_counts[n-1],[0,100]) for n in range(1,m+1)])/1000
#     plt.errorbar(range(1,m+1),mean_counts,yerr = [mean_counts - np.array(CI_counts)[:,0],np.array(CI_counts)[:,1] - mean_counts], fmt = 'o', color = "C1")
        
# plt.savefig("figures/C_test.png", dpi = 400)


n_rows = len(eta_values)
n_cols = len(beta_values)
detail = "l"

I_dist = "Gamma2" 

labels = ["UK LFS " +  r"$(S=1e2)$","UK LFS " + r"$(S=1e3)$", "Split " + r"$(S=1e2)$", "Split " + r"$(S=1e3)$"]
x_tick_deltas = [0.04,0.1,0.3,0.3,0.4]
x_gaps = [0.2,0.5,1.5,1.5,2]
round_is = [2,1,1,1,1]

plot_labels = [["A","B","C","D"],
               ["F","G","H","I"],
               ["K","L","M","N"]]

fig,axes = plt.subplots(n_rows,n_cols, figsize = (20,10), sharey=True,sharex = "col")

fig.supxlabel(r"Base transmission rate ($\beta$)", fontsize = 20)

for i,eta in enumerate(eta_values):
    axes[i,0].set_ylabel(r"$\eta =$" + str(eta), fontsize = 16)
    for j,beta in enumerate(beta_values):
        round_i = round_is[j]
        axes[0,j].set_title(r"$\beta =$" + str(beta), fontsize = 16)
        ax = axes[i,j]
        ax.text(0.01, 0.93, plot_labels[i][j], fontsize=20, transform=ax.transAxes, ha='left', va='top')

        ax.set_ylim(-10,120)
        ax.set_xlim(beta-0.5*x_gaps[j], beta + 3.5*x_gaps[j])

        ax.vlines([beta,beta+x_gaps[j],beta+2*x_gaps[j],beta+3*x_gaps[j]],-5,105,color = "black",linestyle = "--")

        ax.vlines([beta+(k+0.5)*x_gaps[j] for k in range(-1,4)],-100,1000,color = "black")

        ax.set_yticks([])

        x_tick_delta = x_tick_deltas[j]
        ax.set_xticks(np.concat([[beta+k*x_gaps[j]+l*x_tick_delta for l in range(-2,3,1)] for k in range(0,4)]))
        ax.set_xticklabels([round(beta+ k * x_tick_delta,round_i) for k in range(-2,3,1) ]*4,fontsize=12, rotation=90)

        for k,tp in enumerate(hh_dist_prior_tuples):
            if not k in [0,2]:
                print(f"{tp}, beta={beta}, eta={eta}, SAR = {round(100*np.mean(results_SAR[tp][I_dist][beta][eta][detail]),1)}({np.round(100*np.percentile(results_SAR[tp][I_dist][beta][eta][detail],2.5),1)},{np.round(100*np.percentile(results_SAR[tp][I_dist][beta][eta][detail],97.5),1)})%")
            
            CI = results_beta_CI[tp][I_dist][beta][eta][detail]
            means = results_beta_mean[tp][I_dist][beta][eta][detail]
            check = [beta<CI[i][0] or beta>CI[i][1] for i in range(len(CI))]
            error_rate = sum(check)/100
            ax.text(beta+(k)*x_gaps[j],116,labels[k] ,ha='center', va='center',fontsize=8)
            ax.text(beta+(k)*x_gaps[j],109,f" ({round(100*(1-error_rate))}%)",ha='center', va='center',color = (0.75*error_rate,0.75*(1-error_rate),0))
            
            ax.scatter(k*x_gaps[j] + means,np.arange(0,100,1), s=[2 if check[i] else 1 for i in range(len(CI))],
                        color = ['red' if check[i] else 'green' for i in range(len(CI))])
                        
            

            ax.hlines(np.arange(0,100,1),[k*x_gaps[j]+CI[i][0] for i in range(len(CI))], 
                                         [k*x_gaps[j]+CI[i][1] for i in range(len(CI))],
                                         colors = ['red' if check[i] else 'green' for i in range(len(CI))],
                                         linewidth=[0.75 if check[i] else 0.5 for i in range(len(CI))])
            
            ax.vlines((k)*x_gaps[j]+CI[:,0],np.arange(-0.5,99.5,1),np.arange(0.5,100.5,1), 
                        colors = ['red' if check[i] else 'green' for i in range(len(CI))], 
                        linewidth=[0.75 if check[i] else 0.5 for i in range(len(CI))])
            
            ax.vlines((k)*x_gaps[j]+CI[:,1],np.arange(-0.5,99.5,1),np.arange(0.5,100.5,1), 
                        colors = ['red' if check[i] else 'green' for i in range(len(CI))], 
                        linewidth=[0.75 if check[i] else 0.5 for i in range(len(CI))])

plt.tight_layout()
plt.savefig("figures/synthetic_dataset_low_information.png",dpi = 400)