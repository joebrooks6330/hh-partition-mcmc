from HH_case_partition_MCMC import fs_distn_single_type
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")     
plt.rcParams["font.family"] = "monospace"

eta_values = [0,0.5,1]
beta_values = np.arange(0,3,0.1)
beta_values_used = [0.2,0.5,1.5,2.0]
m = 5
hh_sizes = np.arange(2,m+2)

mean_fs_by_size = np.zeros((len(eta_values),len(hh_sizes),len(beta_values)))

example_distns = np.array([[1/2, 0,   0,   0,   1/2],
                          [2/6, 1/6, 0,   1/6, 2/6],
                          [1/5, 1/5, 1/5, 1/5, 1/5]])

mean_fs_example_distn = np.zeros((len(eta_values),len(example_distns),len(beta_values)))

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

fig,axs = plt.subplots(figsize = (20,8),nrows=2,ncols=len(eta_values),sharex = True,sharey="row")

colors = [(1-i/(m),i/(m),0) for i in range(m+1)]
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
        ax2.semilogx(beta_values,mean_fs_by_size[i,j]/n,color = colors[j])
    for j,distn in enumerate(example_distns):
        ax1.semilogx(beta_values,mean_fs_example_distn[i,j],color = "black", linestyle = linestyles[j])
        ax2.semilogx(beta_values,mean_fs_example_distn[i,j]/n,color = "black", linestyle = linestyles[j])
    ax2.set_xticks(beta_values_used)
    ax2.set_xticklabels(beta_values_used)
        
for j,n in enumerate(hh_sizes):
    axs[1,0].plot([0,1],[-10,-10],color = colors[j],label = f"n = {n}")
for j,distn in enumerate(example_distns):
    axs[1,0].plot([0,1],[-10,-10],color = "black",linestyle = linestyles[j],label = str(distn))
    
axs[1,0].legend()
plt.tight_layout()
plt.savefig("figures/final_size_supplementary_fig.png")