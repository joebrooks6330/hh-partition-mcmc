from pickle import load, dump
from os.path import isfile
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from itertools import product
from HH_case_partition_MCMC import IndexChange1dTo2d
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


hh_dist_strings = hh_size_dist_dict.keys()
hh_dist_prior_tuples = [(hh_dist,prior) for hh_dist in hh_dist_strings for prior in [False,True]]
hh_dist_prior_tuples.remove(("UK",True)) # No prior for UK distribution

beta_values = [0.2,0.5,1.5]
eta_values = [0,0.5,1]
detail_values = ["l","m","h"]

results_SAR = {tp: {beta:{eta:{detail: [] for detail in detail_values} for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_beta = {tp: {beta:{eta:{detail: [] for detail in detail_values} for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_llh = {tp: {beta:{eta:{detail: [] for detail in detail_values} for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_beta_mean = {tp: {beta:{eta:{detail: [] for detail in detail_values} for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_beta_CI = {tp: {beta:{eta:{detail: [] for detail in detail_values} for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
params = list(product(hh_dist_prior_tuples,beta_values,eta_values,detail_values))

c=0
for p1 in tqdm(params,desc = "Loading results"):
    tp = p1[0]
    hh_dist = p1[0][0]
    prior = p1[0][1]
    beta = p1[1]
    eta = p1[2]
    detail = p1[3]

    filename = f"outputs/synthetic_data_detail_comparison_{hh_dist}_results"
    if prior:
        filename += "_prior"
    
    filename += f"_beta={beta}_eta={eta}_detail={detail}.pkl"

    if isfile(filename):
        c+=1
        with open(filename, "rb") as f:
            results = load(f)
        C0s = [r[0][0] for r in results]
        m = IndexChange1dTo2d(len(C0s[0])-1)[0]
        dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
        dot_for_cases = np.concatenate([np.arange(0, n + 1) for n in range(1, m + 1)])

        results_SAR[tp][beta][eta][detail] = array([C0.dot(dot_for_cases)/C0.dot(dot_for_contacts) for C0 in C0s])
        results_llh[tp][beta][eta][detail] = array([r[1] for r in results])
        results_beta[tp][beta][eta][detail] = array([r[2] for r in results])
        results_beta_mean[tp][beta][eta][detail] = array([np.mean(r) for r in results_beta[tp][beta][eta][detail]])
        results_beta_CI[tp][beta][eta][detail]= array([np.percentile(r, [2.5,97.5]) for r in results_beta[tp][beta][eta][detail]])



n_rows = len(beta_values)
n_cols = len(eta_values)
detail = "l"

labels = ["UK LFS (2023)", "Split", "Split with prior"]
x_tick_deltas = [0.04,0.1,0.3]
x_gaps = [0.2,0.5,1.5]
round_is = [2,1,1]
fig,axes = plt.subplots(n_rows,n_cols, figsize = (15,10), sharey=True,sharex = "col")

fig.supxlabel(r"Base transmission rate ($\beta$)", fontsize = 20)

for i,eta in enumerate(eta_values):
    axes[i,0].set_ylabel(r"$\eta =$" + str(eta), fontsize = 16)
    for j,beta in enumerate(beta_values):
        round_i = round_is[j]
        axes[0,j].set_title(r"$\beta =$" + str(beta), fontsize = 16)
        ax = axes[i,j]

        ax.set_ylim(-10,120)
        ax.set_xlim(beta-0.5*x_gaps[j], beta + 2.5*x_gaps[j])

        ax.vlines([beta,beta+x_gaps[j],beta+2*x_gaps[j]],-5,105,color = "black",linestyle = "--")

        ax.vlines([beta+(k+0.5)*x_gaps[j] for k in range(-1,3)],-100,1000,color = "black")

        ax.set_yticks([])

        x_tick_delta = x_tick_deltas[j]
        ax.set_xticks(np.concat([[beta+k*x_gaps[j]+l*x_tick_delta for l in range(-2,3,1)] for k in range(0,3)]))
        ax.set_xticklabels([round(beta+ k * x_tick_delta,round_i) for k in range(-2,3,1) ]*3,fontsize=8)

        for k,tp in enumerate(hh_dist_prior_tuples):
            if not tp[1]:
                print(f"{tp}, beta={beta}, eta={eta}, SAR = {round(100*np.mean(results_SAR[tp][beta][eta][detail]),1)}({np.round(100*np.percentile(results_SAR[tp][beta][eta][detail],2.5),1)},{np.round(100*np.percentile(results_SAR[tp][beta][eta][detail],97.5),1)})%")
            
            CI = results_beta_CI[tp][beta][eta][detail]
            means = results_beta_mean[tp][beta][eta][detail]
            check = [beta<CI[i][0] or beta>CI[i][1] for i in range(len(CI))]
            error_rate = sum(check)/100
            ax.text(beta+(k)*x_gaps[j],116,labels[k] ,ha='center', va='center')
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
plt.savefig("figures/synthetic_dataset_low_information_comparison.png",dpi = 400)