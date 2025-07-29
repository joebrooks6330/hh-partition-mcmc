from pickle import load, dump
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


plt.style.use('ggplot')




with open("outputs/single_parameter_fit_params.pkl", "rb") as f:
    params = array(load(f))

beta_values = np.unique(params[:,0])
eta_values = np.unique(params[:,1])
N_hh_values = np.unique(params[:,2])
N_datasets = int(max(params[:,3])+1)
N_chains = int(max(params[:,4])+1)


with open("outputs/single_parameter_fit_results.pkl", "rb") as f:
    results = load(f)

N_particles_per_chain = len(results[0][2])

beta_results = array([r[2] for r in results])

beta_results_dict = {beta:{eta:{N_hh: np.zeros((N_datasets,1+4*N_chains*N_particles_per_chain//5))
                                for N_hh in N_hh_values}
                      for eta in eta_values}
                for beta in beta_values}

i = 0
for beta in beta_values:
    for eta in eta_values:
        for N_hh in N_hh_values:
            for dataset_i in range(N_datasets):
                beta_results_dict[beta][eta][N_hh][dataset_i] = np.concatenate(beta_results[i:i+N_chains,N_particles_per_chain//5:])
                i+= N_chains

KDE_dict = {beta:{eta:{N_hh: [gaussian_kde(beta_results_dict[beta][eta][N_hh][dataset_i]) for dataset_i in range(N_datasets)]
                for N_hh in N_hh_values}
                for eta in eta_values}
                for beta in beta_values}

colours = {beta: (i/(len(beta_values)-1),1 - (i/(len(beta_values)-1)),0) for i,beta in enumerate(beta_values) }

X = np.linspace(0,10,1000)

fig, axs = plt.subplots(len(eta_values), len(N_hh_values), figsize=(20, 15),sharex=True)

for i, eta in enumerate(eta_values):
    for j, N_hh in enumerate(N_hh_values):
        ax = axs[i, j]
        ax.set_xlim(0, 3)
        for beta in beta_values:
            ax.vlines(beta,0,10,color=colours[beta],linestyle="--",label=f'beta={beta}')
            for dataset_i in range(N_datasets):
                ax.plot(X,KDE_dict[beta][eta][N_hh][dataset_i](X),color = colours[beta],linestyle = "--")
                #ax.hist(
                #    beta_results_dict[beta][eta][N_hh][dataset_i],
                #    bins=50,
                #    alpha=0.3,
                #    label=f'beta={beta}, dataset={dataset_i}',
                #    density=True,
                #    color = colours[beta]
                #)
            
        ax.set_xlabel('Beta Value')
        ax.set_ylabel('Density')
        ax.set_title(f'eta={eta}, N_hh={N_hh}')

fig.suptitle('Single Parameter Fits: Beta Histograms by Eta and N_hh', fontsize=18)
fig.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.savefig("figures/single_parameter_fits_histograms.png")
            