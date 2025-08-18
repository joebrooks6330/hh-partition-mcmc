from pickle import load, dump
from os.path import isfile,isdir, getsize
from os import mkdir
from numpy import array
import numpy as np
import matplotlib.pyplot as plt


params_fn = "outputs/n_moves_test_UK_params.pkl"
results_fn = "outputs/n_moves_test_UK_results.pkl"

if not(isfile(params_fn) and isfile(results_fn)):
    print("Files don't exist")
    exit()
else:
    with open(params_fn, "rb") as f:
        params = array(load(f))
    with open(results_fn, "rb") as f2:
        results = load(f2)


beta_results = array([r[2] for r in results])
n_moves_values = np.unique(params[:,0])
N_datasets = max(params[:,1])+1
N_chains = max(params[:,2])
N_particles_per_chain = beta_results.shape[1]

results_beta_dict = {n_moves: np.zeros((N_datasets,4*N_chains*(N_particles_per_chain//5))) for n_moves in n_moves_values}

i = 0
for n_moves in n_moves_values:
    for dataset_i in range(N_datasets):
        results_beta_dict[n_moves][dataset_i] = np.concat(beta_results[i:i+N_chains,1+N_particles_per_chain//5:])
        i+= N_chains


colours = {n_moves: (i/(len(n_moves_values)-1),1 - (i/(len(n_moves_values)-1)),0) for i,n_moves in enumerate(n_moves_values) }

fig,axs = plt.subplots(5,2,figsize = (50,20))
for dataset_i in range(N_datasets):
    ax = axs[dataset_i%5,dataset_i//5]
    for n_moves in n_moves_values[:-1]:
        ax.hist(results_beta_dict[n_moves][dataset_i],bins=np.linspace(0.4,0.6,200),alpha = 1,color = colours[n_moves],label = n_moves,density = True)
axs[0,0].legend()

fn = "figures/n_moves_test_plot.png"
if isdir("figures"):
    plt.savefig(fn,dpi=500)
else:
    mkdir("figures")
    plt.savefig(fn,dpi=500)
print("Figure saved to ", fn)