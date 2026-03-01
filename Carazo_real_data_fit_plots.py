# region Imports
from pickle import load,dump
from os.path import isfile
import matplotlib.pyplot as plt
from seaborn import kdeplot,heatmap
import numpy as np
from numpy import genfromtxt
from HH_case_partition_MCMC import IndexChange1dTo2d,IndexChange2dTo1d, fs_distn_single_type
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import beta,gaussian_kde
from tqdm import tqdm

plt.style.use("ggplot")
plt.rcParams["font.family"] = "monospace"
# endregion

# region Load data and calculate observed SAR
data_fn = "CARAZO_2021_FCDATASET.csv"
dataset_H = np.array(genfromtxt("datasets\\" + data_fn,delimiter=','))
k_max = len(dataset_H)-1
m = IndexChange1dTo2d(k_max)[0]
dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])
dot_for_cases = np.concatenate([np.arange(0, n + 1) for n in range(1, m + 1)])

#Bootstrapping CIs for observed SAR
number_of_cases_list_by_size = [np.concatenate([[y]*int(dataset_H[IndexChange2dTo1d(n,y)])
                                                for y in range(n+1)]) 
                                for n in range(1,m+1)]

N_bootstrap = 1000
SAR_bootstrap = np.zeros((m,N_bootstrap))
for n in range(1,m+1):
    n_cases_list = number_of_cases_list_by_size[n-1]
    for i in range(N_bootstrap):
        outcomes_sample = np.random.choice(n_cases_list,size=len(n_cases_list),replace=True)
        SAR_bootstrap[n-1][i] = sum(outcomes_sample)/(n*len(outcomes_sample))
bootstrap_CI = np.percentile(SAR_bootstrap,[2.5,97.5],axis=1).T

outcomes_list = np.concatenate([[k]*int(nk) for k,nk in enumerate(dataset_H)])
overall_bootstrap_SAR = np.zeros(N_bootstrap)
for i in range(N_bootstrap):
    outcomes_sample = np.random.choice(outcomes_list,size=int(sum(dataset_H)),replace=True)
    outcomes_sample_counts = np.array([sum(outcomes_sample==k) for k in range(len(dataset_H))])
    sample_cases = outcomes_sample_counts.dot(dot_for_cases)
    sample_contacts = outcomes_sample_counts.dot(dot_for_contacts)
    overall_bootstrap_SAR[i] = sample_cases/sample_contacts
overall_bootstrap_CI = np.percentile(overall_bootstrap_SAR,[2.5,97.5])





A = dataset_H*dot_for_cases
B = dataset_H*dot_for_contacts

SAR_overall = sum(A)/sum(B)
SAR_overall_binom_CI = proportion_confint(sum(A),sum(B))

cases_by_size = np.array([sum(A*(dot_for_contacts==n)) for n in range(1,m+1)])
contacts_by_size = np.array([sum(B*(dot_for_contacts==n)) for n in range(1,m+1)])
hh_size_dist_data = np.array([sum(dataset_H*(dot_for_contacts==n)) for n in range(1,m+1)])
SAR_by_size  = np.array([cases_by_size[i]/contacts_by_size[i] for i in range(m)])
SAR_by_size_binom_CI =  np.array([proportion_confint(cases_by_size[i],contacts_by_size[i]) for i in range(m)])
# endregion


#region Set Infectious Period Assumption and load MCMC results
infectious_period_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                                     "Markov": lambda t: 1/(1+t),
                                     "Gamma2": lambda t: (1+(t/2))**(-2)}


inf_period_str = "Gamma2" 
print("Using infectious period assumption:", inf_period_str,end = "\n\n\n")
phi = infectious_period_assumption_dict[inf_period_str]

results_fn_L = "outputs\\"  + data_fn.split(".")[0] + "_low_info_results_" + inf_period_str + ".pkl"
results_fn_M = "outputs\\"  + data_fn.split(".")[0] + "_medium_info_results_" + inf_period_str + ".pkl"
results_fn_H = "outputs\\"  + data_fn.split(".")[0] + "_high_info_results_" + inf_period_str + ".pkl"


# Low information results
if isfile(results_fn_L):
    print("Low info MCMC has already been run for " + data_fn + "\nLoading results...\n")
    with open(results_fn_L,"rb") as f:
        low_info_results = load(f)
else:
    print("Low info MCMC results missing for " + data_fn + ". Please run script.")
    pass
burn_in = len(low_info_results[0])//10
beta_results_L = low_info_results[2][burn_in:]
eta_results_L = low_info_results[3][burn_in:]
C_results_L = low_info_results[0][burn_in:]

# hh_sizes_dist_L = np.array([sum(C_results_L[i]*(dot_for_contacts==n)) for i in range(len(C_results_L)) for n in range(1,m+1)]).reshape(len(C_results_L),m)

# alpha = np.array([41.23596567, 20.02604335, 24.77796057, 8.72501901, 5.23501141])
# alpha_plot = sum(C_results_L[0])*alpha/(sum(alpha))
# plt.plot(hh_sizes_dist_L)
# plt.hlines(alpha_plot,0,len(C_results_L),linestyle="--")
# plt.show()
# # plt.plot(beta_results_L)
# # plt.show()
# quit()
# Medium information results
if isfile(results_fn_M):
    print("Medium info MCMC has already been run for " + data_fn + "\nLoading results...\n")
    with open(results_fn_M,"rb") as f:
        medium_info_results = load(f)
else:
    print("Medium info MCMC results missing for " + data_fn + ". Please run script.")
    quit()

burn_in = len(medium_info_results[0])//5
beta_results_M = medium_info_results[2][burn_in:]
eta_results_M = medium_info_results[3][burn_in:]
C_results_M = medium_info_results[0][burn_in:]


# High information results
if isfile(results_fn_H):
    print("High info MCMC has already been run for " + data_fn + "\nLoading results...\n")
    with open(results_fn_H,"rb") as f:
        high_info_results = load(f)
else:
    print("High info MCMC results missing for " + data_fn + ". Please run script.")
    quit()



burn_in = len(high_info_results[0])//5
beta_results_H = high_info_results[2][burn_in:]
eta_results_H = high_info_results[3][burn_in:]
C_results_H = high_info_results[0][burn_in:]

eta_posterior_fn = "outputs\\eta_kde_posterior_Carazo_high_info_" + inf_period_str + ".pkl"
if isfile(eta_posterior_fn):
    print("Eta posterior KDE already computed")
else:
    print("Computing eta posterior KDE")
    eta_posterior = gaussian_kde(eta_results_H)
    with open(eta_posterior_fn,"wb") as f:
        dump(eta_posterior,f)

# endregion



# region Plotting set up

#Set up data structures for plotting
information_strs = ["Low","Medium","High"]
colors = {"Low":(1,0.3,0.3),"Medium":(0.3,1,0.3),"High":(0.3,0.3,1)}
beta_results = {"Low": beta_results_L, "Medium": beta_results_M, "High": beta_results_H}
eta_results = {"Low": eta_results_L, "Medium": eta_results_M, "High": eta_results_H}
C_results = {"Low": C_results_L, "Medium": C_results_M, "High": C_results_H}

#Set up data structures for other outputs for plotting
SAR_overall_samples = {}
SAR_estimate_by_size_samples = {}
household_size_dist_weighted_samples = {}
household_size_dist_samples = {}

for info in information_strs:
    i_samples = np.random.randint(len(beta_results[info]),size=1000)
    household_size_dist_weighted_samples[info] = [[n*sum(C_results[info][i]*(dot_for_contacts==n))  for n in range(1,m+1)] for i in i_samples]
    household_size_dist_samples[info] = [[sum(C_results[info][i]*(dot_for_contacts==n))  for n in range(1,m+1)] for i in i_samples]
    SAR_estimate_by_size_samples[info] = [[fs_distn_single_type(n,1,beta_results[info][i]/(n**eta_results[info][i]),phi).dot(np.arange(n+1))/n for n in range(1,m+1)] for i in i_samples]
    SAR_overall_samples[info] = [np.dot(hh_dist/sum(hh_dist),SAR_by_size_est) for hh_dist,SAR_by_size_est in zip(household_size_dist_weighted_samples[info],SAR_estimate_by_size_samples[info]) ]
# endregion

#region Plot Scatter 
fig = plt.figure(figsize=(20, 12))
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)
ax4 = plt.subplot(2,1,2)
axes = [ax1, ax2, ax3, ax4]
plot_labels = ["A","B","C","D"]

fig.suptitle(r"Posteriors on $\theta$ - Carazo et al. (2021) ",fontsize = 40)
ax1.set_ylabel(r"Density mixing parameter - $\eta$",fontsize = 20,labelpad=10)

ax1.scatter(beta_results_L,eta_results_L,s=.5,color = colors["Low"],alpha=0.1)
ax1.scatter(np.mean(beta_results_L),np.mean(eta_results_L), s= 50, marker = "x",color = "black")
ax1.axvline(np.mean(beta_results_L), color="black", linestyle=":", alpha=0.7)
ax1.axhline(np.mean(eta_results_L), color="black", linestyle=":", alpha=0.7)
ax1.set_title("Low information",fontsize=20)

ax2.scatter(beta_results_M,eta_results_M,s=0.5,color = colors["Medium"],alpha=0.1)
ax2.scatter(np.mean(beta_results_M),np.mean(eta_results_M), s= 50, marker = "x",color = "black")
ax2.axvline(np.mean(beta_results_M), color="black", linestyle=":", alpha=0.7)
ax2.axhline(np.mean(eta_results_M), color="black", linestyle=":", alpha=0.7)
ax2.set_title("Medium Information",fontsize=20)

ax3.scatter(beta_results_H,eta_results_H,s=0.5,color = colors["High"],alpha=0.1)
ax3.scatter(np.mean(beta_results_H),np.mean(eta_results_H), s= 50, marker = "x",color = "black")
ax3.axvline(np.mean(beta_results_H), color="black", linestyle=":", alpha=0.7)
ax3.axhline(np.mean(eta_results_H), color="black", linestyle=":", alpha=0.7)
ax3.set_title("High Information",fontsize=20)

# Print summary statistics
for info in information_strs:
    beta_mean = np.mean(beta_results[info])
    beta_ci = np.percentile(beta_results[info], [2.5, 97.5])
    eta_mean = np.mean(eta_results[info])
    eta_ci = np.percentile(eta_results[info], [2.5, 97.5])
    
    print(f"\n{info} Information:")
    print(f"  β:   {beta_mean:.4f} (95% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}])")
    print(f"  η:   {eta_mean:.4f} (95% CI: [{eta_ci[0]:.4f}, {eta_ci[1]:.4f}])")

for i,ax in enumerate(axes):
    if i<3:
        ax.set_ylim(-0.05,1.05)
        ax.set_xlim(0.,1.05)
        ax.set_xlabel(r"Base transmission rate - $\beta$",fontsize = 20)
        ax.text(0.025,0.925,plot_labels[i],fontsize = 30)
    else:
        ax.text(0.55,0.475,plot_labels[i],fontsize = 30)

x_pos_all = np.arange(1,m+2,1)
x_pos = x_pos_all[1:]
ax4.set_ylim(0.15,0.55)
ax4.scatter(x = [1-0.09375],y = [SAR_overall], color = 'black',marker='o') 
ax4.errorbar(x_pos+0.0625, SAR_by_size, yerr = [SAR_by_size- bootstrap_CI[:,0] ,  bootstrap_CI[:,1]-SAR_by_size],
              fmt = 'o', color = 'black', ecolor = 'black', elinewidth = 3, capsize = 5, label = "Observed SAR")

ax4_hh_size = ax4.twinx()
hh_size_dist_sample_mean = np.mean(household_size_dist_samples["Low"],axis=0)
hh_size_dist_sample_CI = np.percentile(household_size_dist_samples["Low"],[2.5,97.5],axis=0)
yerr = [hh_size_dist_sample_mean-hh_size_dist_sample_CI[0,:],hh_size_dist_sample_CI[1,:]- hh_size_dist_sample_mean]
ax4_hh_size.bar(x = x_pos-0.25-0.0625, height = hh_size_dist_sample_mean, yerr = yerr,color = colors["Low"],width = 0.125,capsize=5,label = "Estimated size dist. (low info)")
ax4_hh_size.bar(x = x_pos-0.25+0.0625, height = hh_size_dist_data,color = 'black',width = 0.125,label = "Observed size dist.")
ax4_hh_size.legend(fontsize=12, loc = "upper right")
ax4_hh_size.set_yticks(np.arange(0,1450,200))
ax4_hh_size.set_ylim(0,1600)
ax4_hh_size.set_ylabel("Number of house holds",rotation=270, fontsize=20,labelpad=25)
ax4_hh_size.grid(False)

for i,info in enumerate(information_strs):
    SAR_estimate_overall_mean = np.mean(SAR_overall_samples[info])
    SAR_estimate_overall_CI = np.percentile(SAR_overall_samples[info],[2.5,97.5])
    ax4.errorbar(x = [1+0.09375+0.0625*(i)], y = [np.mean(SAR_overall_samples[info])],
                yerr = [[SAR_estimate_overall_mean-SAR_estimate_overall_CI[0]],[SAR_estimate_overall_CI[1]- SAR_estimate_overall_mean]],
                fmt = 'o',color  = colors[info], elinewidth = 3, capsize = 5)
    
    SAR_estimate_by_size_sample_mean = np.mean(SAR_estimate_by_size_samples[info],axis=0)
    SAR_estimate_by_size_sample_CI = np.percentile(SAR_estimate_by_size_samples[info],[2.5,97.5],axis=0)
    
    yerr = [SAR_estimate_by_size_sample_mean-SAR_estimate_by_size_sample_CI[0,:],SAR_estimate_by_size_sample_CI[1,:]- SAR_estimate_by_size_sample_mean]
    ax4.errorbar(x = x_pos+0.25+0.0625*(i), y = SAR_estimate_by_size_sample_mean, yerr = yerr,
                  fmt = 'o', color = colors[info], ecolor = colors[info], elinewidth = 3, capsize = 5, label = info + " info SAR est.")


ax4.set_xlim(0.5,m+1.5)
ax4.set_xticks(x_pos_all)
ax4.set_xticklabels(["Overall"] + list(x_pos),fontsize=15)
ax4.set_xlabel("Household Size",fontsize=20)
ax4.set_ylabel("Secondary Attack Rate",fontsize=20)
ax4.set_ylim(0.15,0.5125)
ax4.set_yticks(np.arange(0.15,0.55,0.05))
ax4.legend(fontsize=12, loc = (0.,0.6))
ax4.vlines(x_pos-0.5,0,1,color = "grey",linestyle = "--")
plt.savefig("figures\\CARAZO_scatterplot_"+ inf_period_str + ".png",bbox_inches='tight', dpi = 400)
# endregion




quit() #Remove for supplementary plots
#Supplementary Plots

# region KDE plot
fig = plt.figure(figsize=(20, 12))
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)
ax4 = plt.subplot(2,1,2)
axes = [ax1, ax2, ax3, ax4]
plot_labels = ["A","B","C","D"]

fig.suptitle(r"Posteriors on $\theta$ - Carazo et al. (2021) ",fontsize = 40)
ax1.set_ylabel(r"Density mixing parameter - $\eta$",fontsize = 20,labelpad=10)
kdeplot(x = beta_results_L,y = eta_results_L,ax = ax1,color = colors["Low"],levels=5,bw_adjust=2)
ax1.scatter(np.mean(beta_results_L),np.mean(eta_results_L), s= 50, marker = "x",color = "black")
ax1.axvline(np.mean(beta_results_L), color="black", linestyle=":", alpha=0.7)
ax1.axhline(np.mean(eta_results_L), color="black", linestyle=":", alpha=0.7)
ax1.set_title("Low information",fontsize=20)

kdeplot(x = beta_results_M,y = eta_results_M,ax = ax2,color = colors["Medium"],levels=5,bw_adjust=2)
ax2.scatter(np.mean(beta_results_M),np.mean(eta_results_M), s= 50, marker = "x",color = "black")
ax2.axvline(np.mean(beta_results_M), color="black", linestyle=":", alpha=0.7)
ax2.axhline(np.mean(eta_results_M), color="black", linestyle=":", alpha=0.7)
ax2.set_title("Medium Information",fontsize=20)

kdeplot(x = beta_results_H,y = eta_results_H,ax = ax3,color = colors["High"],levels=5,bw_adjust=2)
ax3.scatter(np.mean(beta_results_H),np.mean(eta_results_H), s= 50, marker = "x",color = "black")
ax3.axvline(np.mean(beta_results_H), color="black", linestyle=":", alpha=0.7)
ax3.axhline(np.mean(eta_results_H), color="black", linestyle=":", alpha=0.7)
ax3.set_title("High Information",fontsize=20)

for i,ax in enumerate(axes):
    if i<3:
        ax.set_ylim(-0.05,1.05)
        ax.set_xlim(0.,0.8)
        ax.set_xlabel(r"Base transmission rate - $\beta$",fontsize = 20)
        ax.text(0.025,0.925,plot_labels[i],fontsize = 30)
    else:
        ax.text(0.55,0.475,plot_labels[i],fontsize = 30)

x_pos_all = np.arange(1,m+2,1)
x_pos = x_pos_all[1:]
ax4.set_ylim(0.15,0.55)
ax4.scatter(x = [1-0.09375],y = [SAR_overall], color = 'black',marker='o') 
ax4.errorbar(x_pos+0.0625, SAR_by_size, yerr = [SAR_by_size- bootstrap_CI[:,0] ,  bootstrap_CI[:,1]-SAR_by_size],
              fmt = 'o', color = 'black', ecolor = 'black', elinewidth = 3, capsize = 5, label = "Observed SAR")

ax4_hh_size = ax4.twinx()
hh_size_dist_sample_mean = np.mean(household_size_dist_samples["Low"],axis=0)
hh_size_dist_sample_CI = np.percentile(household_size_dist_samples["Low"],[2.5,97.5],axis=0)
yerr = [hh_size_dist_sample_mean-hh_size_dist_sample_CI[0,:],hh_size_dist_sample_CI[1,:]- hh_size_dist_sample_mean]
ax4_hh_size.bar(x = x_pos-0.25-0.0625, height = hh_size_dist_sample_mean, yerr = yerr,color = colors["Low"],width = 0.125,capsize=5,label = "Estimated size dist. (low info)")
ax4_hh_size.bar(x = x_pos-0.25+0.0625, height = hh_size_dist_data,color = 'black',width = 0.125,label = "Observed size dist.")
ax4_hh_size.legend(fontsize=12, loc = "upper right")
ax4_hh_size.set_yticks(np.arange(0,1450,200))
ax4_hh_size.set_ylim(0,1450)
ax4_hh_size.set_ylabel("Number of households",rotation=270, fontsize=20,labelpad=25)
ax4_hh_size.grid(False)

for i,info in enumerate(information_strs):
    SAR_estimate_overall_mean = np.mean(SAR_overall_samples[info])
    SAR_estimate_overall_CI = np.percentile(SAR_overall_samples[info],[2.5,97.5])
    ax4.errorbar(x = [1+0.09375+0.0625*(i)], y = [np.mean(SAR_overall_samples[info])],
                 yerr = [[SAR_estimate_overall_mean-SAR_estimate_overall_CI[0]],[SAR_estimate_overall_CI[1]- SAR_estimate_overall_mean]],
                 fmt = 'o',color  = colors[info], elinewidth = 3, capsize = 5)
    
    SAR_estimate_by_size_sample_mean = np.mean(SAR_estimate_by_size_samples[info],axis=0)
    SAR_estimate_by_size_sample_CI = np.percentile(SAR_estimate_by_size_samples[info],[2.5,97.5],axis=0)
    
    yerr = [SAR_estimate_by_size_sample_mean-SAR_estimate_by_size_sample_CI[0,:],SAR_estimate_by_size_sample_CI[1,:]- SAR_estimate_by_size_sample_mean]
    ax4.errorbar(x = x_pos+0.25+0.0625*(i), y = SAR_estimate_by_size_sample_mean, yerr = yerr,
                  fmt = 'o', color = colors[info], ecolor = colors[info], elinewidth = 3, capsize = 5, label = info + " info SAR est.")


ax4.set_xlim(0.5,m+1.5)
ax4.set_xticks(x_pos_all)
ax4.set_xticklabels(["Overall"] + list(x_pos),fontsize=15)
ax4.set_xlabel("Household Size",fontsize=20)
ax4.set_ylabel("Secondary Attack Rate",fontsize=20)
ax4.set_ylim(0.15,0.5125)
ax4.set_yticks(np.arange(0.15,0.55,0.05))
ax4.legend(fontsize=12, loc = (0.,0.6))
ax4.vlines(x_pos-0.5,0,1,color = "grey",linestyle = "--")
plt.savefig("figures\\CARAZO_kdeplot.png",bbox_inches='tight', dpi = 400)
# endregion 

# region Load Low info (gaussian(x,0.1) prior on eta)
eta_values = [0,0.5,0.7,1]

beta_results_LF = {eta: [] for eta in eta_values}
eta_results_LF = {eta: [] for eta in eta_values}
beta_results_LF_95CI = np.zeros((len(eta_values),2))

for i,eta in enumerate(eta_values):
    results_fn_L_eta = "outputs\\" + data_fn.split(".")[0] + "_low_info_results_eta=" + str(eta) +"_prior.pkl"
    if isfile(results_fn_L_eta):
        print("Low info MCMC for fixed eta = " + str(eta) + " has already been run for " + data_fn + "\nLoading results...")
        with open(results_fn_L_eta,"rb") as f:
            low_info_results = load(f)
        burn_in = len(low_info_results[0])//10
        beta_results_LF[eta] = low_info_results[2][burn_in:]
        eta_results_LF[eta] = low_info_results[3][burn_in:]
        #beta_results_LF_95CI[i] = np.percentile(beta_results_LF[eta],[2.5,97.5])
    else:
        print("Low info MCMC results missing for fixed eta = " + str(eta) + " for " + data_fn + ". Please run script.")
        quit()
# endregion

# region Load Low info (prior on household size distribution, eta not fixed)
results_fn_L_priored = "outputs\\" + data_fn.split(".")[0] + "_low_info_size_dist_priored_results.pkl"
if isfile(results_fn_L_priored):
    print("Low info (with prior) MCMC has already been run for " + data_fn + "\nLoading results...")
    with open(results_fn_L_priored,"rb") as f:
        low_info_results = load(f)
else:
    print("Low info (with prior) MCMC results missing for " + data_fn + ". Please run script.")
    pass
burn_in = len(low_info_results[0])//10
beta_results_L_priored = low_info_results[2][burn_in:]
eta_results_L_priored = low_info_results[3][burn_in:]
# endregion

# region Plot Low info test for different priors on eta/size dist
fig1,axs = plt.subplots(3,2,sharex=True,sharey=True,figsize=(10,10))
plot_labels = ["A","B","C","D","E","F"]

fig1.suptitle("Posteriors for low information case\nCarazo et al. (2021)",fontsize=20,y=0.99)
fig1.supxlabel(r"Base transmission rate ($\beta$)",fontsize=16,y=0.05)
fig1.supylabel(r"Density mixing parameter ($\eta$)",fontsize=16,x=0.03)

axs[0][0].set_title("No prior")
axs[0][0].scatter(beta_results_L,eta_results_L,s=0.5,alpha = 0.25,color = "r")
axs[0][0].text(0.15,1.25,plot_labels[0],fontsize=20)

axs[0][1].set_title("Prior on size dist.")
axs[0][1].scatter(beta_results_L_priored,eta_results_L_priored,s=0.5,color = "blue",alpha=0.25)
axs[0][1].text(0.15,1.25,plot_labels[1],fontsize=20)

for i,eta in enumerate(eta_values):
    x = 1 + i//2
    y = i%2
    axs[x][y].set_title("Priored eta centered at " + str(eta))
    axs[x][y].scatter(beta_results_LF[eta],eta_results_LF[eta],s=.5,color = "green",alpha=0.5)
    axs[x][y].text(0.15,1.25,plot_labels[i+2],fontsize=20)

plt.savefig("figures/low_info_scatter_priors_comparison.png",bbox_inches='tight', dpi = 400)

# endregion