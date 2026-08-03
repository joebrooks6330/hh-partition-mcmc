from HH_case_partition_MCMC import fs_distn_single_type
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile
from pickle import load
plt.style.use("ggplot")     
plt.rcParams["font.family"] = "monospace"
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

#hh_dist_mcmc_mean = np.array([3.79054769e-01, 1.98325209e-04, 5.78735472e-01, 2.24500312e-03, 3.97664304e-02])#mean for eta = 1 beta = 0.5


eta_values = [0,0.5,1]
beta_values = np.arange(0,3,0.01)
beta_values_used = [0.2,0.5,1.5]
m = 5
hh_sizes = np.arange(2,m+2)

mean_fs_by_size = np.zeros((len(eta_values),len(hh_sizes),len(beta_values)))

example_distns = np.array([hh_dist_split_weighted,hh_dist_UK_weighted])

mean_fs_example_distn = np.zeros((len(eta_values),len(example_distns),len(beta_values)))

mean_SAR_example_distn = np.zeros((len(eta_values),len(example_distns),len(beta_values)))

infectious_period_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                                     "Markov": lambda t: 1/(1+t),
                                     "Gamma2": lambda t: (1+(t/2))**(-2)}

inf_period_str = "Gamma2" 
phi = infectious_period_assumption_dict[inf_period_str]

for i,eta in enumerate(eta_values):
    for j,n in enumerate(hh_sizes):
        dotv = np.arange(0,n)
        for k,beta in enumerate(beta_values):
            fs = fs_distn_single_type(int(n-1),1,beta/(n-1)**eta,phi)
            mean_fs_by_size[i,j,k] = fs.dot(dotv)
    fs_by_size = mean_fs_by_size[i].T
    for j,distn in enumerate(example_distns):
        mean_fs_example_distn[i,j] = fs_by_size.dot(distn)
        mean_SAR_example_distn[i,j] = (fs_by_size.dot(distn))/(split_mu-1)
fig,axs = plt.subplots(figsize = (20,8),nrows=2,ncols=len(eta_values),sharex = True,sharey="row")

colors = [(1-i/(m-1),i/(m-1),0) for i in range(m)]
linestyles = ["-","--",":"]
for i,eta in enumerate(eta_values):
    ax1 = axs[0,i]
    ax1.set_title(r"$\eta = $" +str(eta),fontsize =25)
    ax1.set_xlim(1e-1,3)
    ax1.set_ylim(0,5)
    ax1.vlines(beta_values_used,-1,6,color = "black",linestyle = "--")
    
    ax2 = axs[1,i]
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylim(0,1)
    ax2.set_xlim(1e-1,3)
    ax2.vlines(beta_values_used,-1,2,color = "black",linestyle = "--")
    
    if i==0:
        ax2.set_ylabel("SAR",fontsize  =15)
        ax1.set_ylabel("Final Size",fontsize  =15)
    for j,n in enumerate(hh_sizes):
        ax1.semilogx(beta_values,mean_fs_by_size[i,j],color = colors[j])
        ax2.semilogx(beta_values,mean_fs_by_size[i,j]/(n-1),color = colors[j])
    for j,distn in enumerate(example_distns):
        ax1.semilogx(beta_values,mean_fs_example_distn[i,j],color = "black", linestyle = linestyles[j])
        ax2.semilogx(beta_values,mean_SAR_example_distn[i,j],color = "black", linestyle = linestyles[j])
    ax2.set_xticks(beta_values_used)
    ax2.set_xticklabels(beta_values_used)
        
for j,n in enumerate(hh_sizes):
    axs[1,0].plot([0,1],[-10,-10],color = colors[j],label = f"Size = {n}")
    
labels = ["Split","UK LFS"]
for j,distn in enumerate(example_distns):
    axs[1,2].plot([0,1],[-10,-10],color = "black",linestyle = linestyles[j],label = labels[j])
    
axs[1,0].legend()
axs[1,2].legend()

plt.tight_layout()
plt.savefig("figures/final_size_supplementary_fig.png")

#region Plot: eta effect on transmission rate

eta_values = [0,0.5,1]
beta_value = 1.2
n_values = np.arange(1,6.1,1)
beta_n_values = np.array([beta_value/(n_values)**eta for eta in eta_values])

fig,axs = plt.subplots(1,1,figsize = (20,8))
labels = [r"$\eta = 0$",r"$\eta = 0.5$",r"$\eta = 1$"]
colors = ["blue","red","green"]

for i,eta in enumerate(eta_values):
    axs.bar(n_values-0.25+i*0.25, beta_n_values[i],label = labels[i],width = 0.2,facecolor = colors[i],alpha=0.7)
axs.set_xlabel(r"Number of contacts ($n$)",fontsize = 25)
axs.set_ylabel(r"Transmission rate ($\beta_n$)",fontsize = 25)
axs.tick_params(axis='both', which='major', labelsize=20)
axs.set_ylim(0,1.3)
fig.legend(fontsize=25,loc="upper center",ncol = 3)
plt.savefig("figures/eta_effect_on_beta_n.png",dpi=200)
#endregion

#region Plot: effect of Infectious period assumption on final size

infectious_period_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                                     "Markov": lambda t: 1/(1+t),
                                     "Gamma(2,2)": lambda t: (1+(t/2))**(-2)}

inf_period_strs = ["Fixed","Gamma(2,2)","Markov"]

from scipy.stats import expon,gamma

inf_period_dists = {"Fixed": lambda t: 0 if t!=1 else 10,
                  "Markov": expon(scale=1).pdf,
                  "Gamma(2,2)": gamma(a=2,scale=1).pdf}

fig2,axs2 = plt.subplots(2,3,figsize = (20,10),sharey = "row")
t_values = np.linspace(0,8,100)

n=5
beta = 1.4
eta = 1

for i,inf_period_str in enumerate(inf_period_strs):
    if inf_period_str=="Fixed":
        axs2[0,i].plot(t_values, [inf_period_dists[inf_period_str](t) for t in t_values],color = "black")
        axs2[0,i].vlines([1],0,10,color = "black",linestyle = "--")
    else:
        axs2[0,i].plot(t_values,inf_period_dists[inf_period_str](t_values),color = "black")
    axs2[0,i].set_ylim(-0.02,1.1)
    axs2[0,i].set_xlabel(r"$t$",fontsize = 15)
    axs2[0,0].set_ylabel(r"Infectious period PDF",fontsize = 15)
    axs2[0,i].set_title(inf_period_str + " Infectious Period",fontsize = 15)

    fs = fs_distn_single_type(n,1,beta/(n)**eta,infectious_period_assumption_dict[inf_period_str])
    axs2[1,i].bar(np.arange(0,n+1),fs,color = "black")
    axs2[1,i].set_xlabel(r"Final Size",fontsize = 15)
    axs2[1,0].set_ylabel(r"Final Size Distribution",fontsize = 15)

plt.tight_layout()
plt.savefig("figures/infectious_period_assumptions.png",dpi=200)
    
    
    

