#region imports
from pickle import load, dump
from os.path import isfile
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from itertools import product
from HH_case_partition_MCMC import IndexChange1dTo2d, IndexChange2dTo1d
from partition_functions import FlatPartition, get_simple_dataset
import arviz as az
plt.style.use("ggplot")     
plt.rcParams["font.family"] = "monospace"
#endregion

#region Load and plot HH size dist
# Import household size distributions used to generate synthetic data
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

hh_dist_weighted_dict = {"UK": hh_dist_UK_weighted, "split": hh_dist_split_weighted}

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
#endregion

#region Load synthetic datasets
if isfile("datasets/synthetic_100.pkl"):
    with open("datasets/synthetic_100.pkl", "rb") as f:
        synthetic_datasets = load(f)


hh_dist_strings = hh_size_dist_dict.keys()
hh_dist_prior_tuples = [(hh_dist,alpha_str) for hh_dist in hh_dist_strings for alpha_str in ["false","true"]]
I_dist_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                          "Markov": lambda t: 1/(1+t),
                          "Gamma2": lambda t: (1+(t/2))**(-2)}


I_dist = "Gamma2" 
print("I assumption: " + I_dist)
N_hh = 100
print("N_hh =" + str(N_hh))
plot_HHSD = False
plot_beta_trace = False
plot_ESS = True

beta_values = [0.2,0.5,1.5]
eta_values = [0,0.5,1]
detail = "l"
N_datasets = 100
k_max = int(0.5*(m+3)*(m))

                

results_C = {tp: {beta:{eta: np.zeros((N_datasets)) for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_SAR = {tp: {beta:{eta: np.zeros(N_datasets) for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_beta = {tp: {beta:{eta: np.zeros(N_datasets) for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_llh = {tp: {beta:{eta: np.zeros(N_datasets) for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_beta_mean = {tp: {beta:{eta: np.zeros(N_datasets) for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_beta_CI = {tp: {beta:{eta: np.zeros((N_datasets,2)) for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
results_ESS = {tp: {beta:{eta: np.zeros((N_datasets)) for eta in eta_values} for beta in beta_values} for tp in hh_dist_prior_tuples}
params = list(product(hh_dist_prior_tuples,beta_values,eta_values))

c=0
for p1 in tqdm(params,desc = "Loading results"):
    tp = p1[0]
    hh_dist = p1[0][0]
    alpha_str = p1[0][1]
    beta = p1[1]
    eta = p1[2]
    filename = f"outputs/synth_Nhh={N_hh}/synthetic_data_validation_{hh_dist}_alpha={alpha_str}_I_dist={I_dist}_Nhh={N_hh}_results"
    filename += f"_beta={beta}_eta={eta}_detail={detail}.pkl"

    if isfile(filename):
        c+=1
        datasets = synthetic_datasets[I_dist][beta][eta][N_hh][hh_dist]
        with open(filename, "rb") as f:
            results = load(f)
        m = IndexChange1dTo2d(len(datasets[0])-1)[0]
        n_iters = len(results[0][0])
        dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
        dot_for_cases = np.concatenate([np.arange(0, n + 1) for n in range(1, m + 1)])

        C_results = array([r[0][:] for r in results])
        results_C[tp][beta][eta] = C_results
        beta_results = array([r[2][n_iters//5:] for r in results])
        results_beta[tp][beta][eta] = beta_results
        results_SAR[tp][beta][eta] = array([C.dot(dot_for_cases)/C.dot(dot_for_contacts) for C in datasets])
        
        results_beta_mean[tp][beta][eta] = array([np.mean(r) for r in results_beta[tp][beta][eta]]) #type:ignore
        results_beta_CI[tp][beta][eta]= array([np.percentile(r, [2.5,97.5]) for r in results_beta[tp][beta][eta]]) #type:ignore
        
        ESS_results = np.array([az.ess(np.log(br.reshape(1,-1))) for br in beta_results])
        acceptance_results = [r[-1] for r in results]
        if tp == ("split","true") and beta==1.5:
            fig_temp,ax_temp = plt.subplots(2,1)
            fig_temp.suptitle(tp[0]+tp[1])
            i_w = np.argmin(ESS_results)
            ax_temp[0].plot(beta_results[i_w])
            print(sum(acceptance_results)/len(acceptance_results))
            ax_temp[0].set_title(f"beta = {beta}, eta = {eta}, AR = {acceptance_results[i_w]}, SAR = {results_SAR[tp][beta][eta][i_w]}")
            ax_temp[1].plot(C_results[i_w,:,:2])
        
            print()
            plt.show()
        results_ESS[tp][beta][eta] = ESS_results
        
        # comb_results = np.concatenate([C_results[:,n_iters//5:,:], beta_results[:,:,None]],axis=2)
        # results_ESS[tp][beta][eta] = np.array([az.ess(comb.reshape(1,len(comb),-1)) for comb in comb_results])
            
        
        
        
        
        
    else:
        print(f"File {filename} not found.")
#endregion

#region Plot hh size distn trace plots
if detail == "l" and plot_HHSD:
    for tp in tqdm(hh_dist_prior_tuples, desc="Plotting HHSD traceplots"):
        hhsd_hh_dist_traceplot = tp[0]
        alpha_str = tp[1]

        fig,axes = plt.subplots(len(eta_values),len(beta_values), figsize = (20,10))
        fig.supxlabel("MCMC iteration (thinned)", fontsize = 35)
        fig.supylabel("Total variation distance", fontsize = 35)
        for i,beta in enumerate(beta_values):
            for j,eta in enumerate(eta_values):
                axes[j,0].set_ylabel(r"$\eta$:" + str(eta),fontsize=25)
                axes[0,i].set_title(r"$\beta = $" + str(beta),fontsize=25)
                
                ax = axes[j,i]
                
        
                C_results = results_C[(hhsd_hh_dist_traceplot,alpha_str)][beta][eta]
                if len(C_results.shape)==1:
                    pass
                else:
                    hh_size_counts = np.array([(C_results*(dot_for_contacts==n)).sum(axis=2) for n in range(1,m+1)])
                    total_variation_distance = np.zeros(hh_size_counts.shape[1:])
                    for n in range(1,m+1):
                        total_variation_distance += np.abs(hh_size_counts[n-1]/N_hh - hh_dist_weighted_dict[hhsd_hh_dist_traceplot][n-1])/2
                    ax.plot(total_variation_distance.T,color = "black",alpha = 1,lw=0.05)

        #fig.tight_layout(rect=(0, 0, 0.85, 1))
        fig.savefig(f"figures/validation figures/{hhsd_hh_dist_traceplot}_hh_size_dist_TVD_trace_plot_alpha={alpha_str}_Nhh={N_hh}.png", dpi = 400,bbox_inches ='tight')
        
        #fig2.tight_layout(rect=(0, 0, 0.85, 1))
        
        
        fig2,axes2 = plt.subplots(3,2, figsize = (20,10))
        for n in range(1,m+1):
            colors = [(i/n,(n-i)/n,0.) for i in range(n+1)]
            ax2 = axes2[(n-1)//2,(n-1)%2]
            ax2.set_title(f"n = {n+1}", fontsize = 25)
            s0 = IndexChange2dTo1d(n,0)
            i_w = np.argmin(results_ESS[(hhsd_hh_dist_traceplot,alpha_str)][1.5][0])
            for i in range(n+1):
                if results_C[(hhsd_hh_dist_traceplot,alpha_str)][beta][eta].shape[0]>1:
                    C_results_ex = results_C[(hhsd_hh_dist_traceplot,alpha_str)][1.5][0][i_w,:,s0+i]
                    ax2.plot(C_results_ex,color = colors[i])
            
            
        
        fig2.savefig(f"figures/validation figures/{hhsd_hh_dist_traceplot}_partition_example_trace_plot_alpha={alpha_str}_Nhh={N_hh}.png", dpi = 200,bbox_inches ='tight')

    print(f"HHSD trace plots saved.")   
    
    
#endregion

#region Plot beta trace plots
if plot_beta_trace:
    for tp in tqdm(hh_dist_prior_tuples, desc="Plotting beta traceplots"):
            hhsd_hh_dist_traceplot = tp[0]
            alpha_str = tp[1]

            fig3,axes3 = plt.subplots(len(eta_values),len(beta_values), figsize = (20,10),sharey = True)
            fig3.supxlabel("MCMC iteration (thinned)", fontsize = 35)
            fig3.supylabel(r"$\beta$", fontsize = 35)
            
            for i,beta in enumerate(beta_values):
                for j,eta in enumerate(eta_values):
                    axes3[j,0].set_ylabel(r"$\eta$:" + str(eta),fontsize=25)
                    axes3[0,i].set_title(r"$\beta = $" + str(beta),fontsize=25)
                    
                    ax3 = axes3[j,i]
                    
                    beta_results = results_beta[(hhsd_hh_dist_traceplot,alpha_str)][beta][eta]
                    if len(beta_results.shape)==1:
                        pass
                    else:
                        n_iters = beta_results.shape[1]
                        ax3.plot(beta_results.T,color = "black",alpha = 0.05,lw=0.5)
                        ax3.hlines(beta, 0, 2e8,linestyle = "--",lw = 3,color = "black")
                        ax3.set_xlim(0,1.15*n_iters)
                        ax3.set_ylim(0,1.5*beta)

            fig3.savefig(f"figures/validation figures/{hhsd_hh_dist_traceplot}_beta_trace_plot_alpha={alpha_str}_Nhh={N_hh}.png", dpi = 200,bbox_inches ='tight')
#endregion
       
#region ESS Plots
if plot_ESS:
    for tp in tqdm(hh_dist_prior_tuples, desc="Plotting ESS plots"):
            hhsd_hh_dist_traceplot = tp[0]
            alpha_str = tp[1]

            fig4,axes4 = plt.subplots(len(eta_values),len(beta_values), figsize = (20,10),sharex = True)
            fig4.supxlabel(r"Dataset $i$", fontsize = 35)
            fig4.supylabel(r"$ESS$", fontsize = 35)
            
            for i,beta in enumerate(beta_values):
                for j,eta in enumerate(eta_values):
                    axes4[j,0].set_ylabel(r"$\eta$:" + str(eta),fontsize=25)
                    axes4[0,i].set_title(r"$\beta = $" + str(beta),fontsize=25)
                    
                    ax4 = axes4[j,i]
                    
                    ESS_results = results_ESS[(hhsd_hh_dist_traceplot,alpha_str)][beta][eta]
                    if not sum(ESS_results)==0:
                        ax4.bar(range(N_datasets),np.sort(ESS_results),color = "black")
                        ax4.set_xlim(-1,101)
                        ax4.hlines([200],-10,110)

            fig4.savefig(f"figures/validation figures/{hhsd_hh_dist_traceplot}_ESS_plot_alpha={alpha_str}_Nhh={N_hh}.png", dpi = 200,bbox_inches ='tight')
#endregion

#region Main Plots
n_rows = len(eta_values)
n_cols = len(beta_values)


labels = [r"UK Data, Split $\mathbf{\alpha}$",
          r"UK Data, UK $\mathbf{\alpha}$", 
          r"Split Data, UK $\mathbf{\alpha}$", 
          r"Split Data, Split $\mathbf{\alpha}$"]
x_tick_deltas = {25: [0.1,0.25,0.8,1.0],
                 100:[0.08,0.2,0.6,0.6,0.8],
                 1000:[0.04,0.1,0.3,0.3,0.4]}[N_hh]
x_gaps = {25:[0.6,1.5,4.0,5.0],
          100: [0.4,1.,3.0,3.0,4.0],
          1000: [0.2,0.5,1.5,1.5,2.0]}[N_hh]    
round_is = [2,1,1,1,1]


plot_labels = [["A","B","C","D"],
               ["E","F","G","H"],
               ["I","J","K","L"]]

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

        

        ax.set_yticks([])

        x_tick_delta = x_tick_deltas[j]
        ax.set_xticks(np.concat([[beta+k*x_gaps[j]+l*x_tick_delta for l in range(-2,3,1)] for k in range(0,4)]))
        ax.set_xticklabels([round(beta+ k * x_tick_delta,round_i) for k in range(-2,3,1) ]*4,fontsize=12, rotation=90)

        for k,tp in enumerate(hh_dist_prior_tuples):
            if not k in [0,2]:
                print(f"{tp}, beta={beta}, eta={eta}, SAR = {round(100*np.mean(results_SAR[tp][beta][eta]),1)}({np.round(100*np.percentile(results_SAR[tp][beta][eta],2.5),1)},{np.round(100*np.percentile(results_SAR[tp][beta][eta],97.5),1)})%")
            
            CI = results_beta_CI[tp][beta][eta]
            means = results_beta_mean[tp][beta][eta]
            overall_mean = np.mean(means)
            ax.vlines([k*x_gaps[j] + overall_mean], -5, 105, color = "black", linestyle = "-", alpha = 0.5)
            check = [beta<CI[i][0] or beta>CI[i][1] for i in range(len(CI))]
            check2 = [CI[i][1]>beta+0.5*x_gaps[j] for i in range(len(CI))]
            error_rate = sum(check)/100
            ax.text(beta+(k)*x_gaps[j],116,labels[k] ,ha='center', va='center',fontsize=8)
            ax.text(beta+(k)*x_gaps[j],109,f" ({round(100*(1-error_rate))}%)",ha='center', va='center',color = (0.75*error_rate,0.75*(1-error_rate),0))
            
            ax.scatter(k*x_gaps[j] + means,np.arange(0,100,1), s=[2 if check[i] else 1 for i in range(len(CI))],
                        color = ['red' if check[i] else 'green' for i in range(len(CI))],
                        alpha = [0 if check2[i] else 1 for i in range(len(CI))])
                        
            

            ax.hlines(np.arange(0,100,1),[k*x_gaps[j]+CI[i][0] for i in range(len(CI))], 
                                         [k*x_gaps[j]+min(CI[i][1],beta+0.5*x_gaps[j]) for i in range(len(CI))],
                                         colors = ['red' if check[i] else 'green' for i in range(len(CI))],
                                         linewidth=[0.75 if check[i] else 0.5 for i in range(len(CI))])
            
            ax.vlines((k)*x_gaps[j]+CI[:,0],np.arange(-0.5,99.5,1),np.arange(0.5,100.5,1), 
                        colors = ['red' if check[i] else 'green' for i in range(len(CI))], 
                        linewidth=[0.75 if check[i] else 0.5 for i in range(len(CI))])
            
            ax.vlines((k)*x_gaps[j]+CI[:,1],np.arange(-0.5,99.5,1),np.arange(0.5,100.5,1), 
                        colors = ['red' if check[i] else 'green' for i in range(len(CI))], 
                        linewidth=[0.75 if check[i] else 0.5 for i in range(len(CI))],
                        alpha = [0 if check2[i] else 1 for i in range(len(CI))])
            
        ax.vlines([beta+(k+0.5)*x_gaps[j] for k in range(-1,4)],-100,1000,color = "black")
plt.tight_layout()
plt.savefig(f"figures/validation figures/synthetic_dataset_low_information_comparison_{I_dist}_Nhh={N_hh}_{detail}.png",dpi = 400)
#endregion