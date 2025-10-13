from pickle import load, dump
from os.path import isfile,isdir, getsize
from os import mkdir
from numpy import array
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


params_fn = "outputs/p_beta_move_test_UK_params.pkl"
results_fn = "outputs/p_beta_move_test_UK_results.pkl"

if not(isfile(params_fn) and isfile(results_fn)):
    print("Files don't exist")
    exit()
else:
    with open(params_fn, "rb") as f:
        params = array(load(f))
    with open(results_fn, "rb") as f2:
        results = load(f2)


beta_results = array([r[2] for r in results])
p_beta_move_values = np.unique(params[:,0])
N_datasets = int(max(params[:,1]))+1
N_chains = int(max(params[:,2]))+1
N_particles_per_chain = beta_results.shape[1]

results_beta_dict = {p_beta_move: np.zeros((N_datasets,4*N_chains*(N_particles_per_chain//5))) for p_beta_move in p_beta_move_values}
results_beta_95CI_dict = {p_beta_move:np.zeros((N_datasets,(N_particles_per_chain-100)//10,2)) for p_beta_move in p_beta_move_values}
i = 0
for p_beta_move in p_beta_move_values:
    for dataset_i in range(N_datasets):
        results_beta_dict[p_beta_move][dataset_i] = np.concat(beta_results[i:i+N_chains,1+N_particles_per_chain//5:])
        
        for k in tqdm(range(1,(N_particles_per_chain-100)//10),mininterval=10,leave=False,desc = f"p_beta_move = {p_beta_move}, Dataset {dataset_i}"):
            trunc_results = beta_results[i:i+N_chains,100:100+k*10].flatten()
            results_beta_95CI_dict[p_beta_move][dataset_i][k] = np.percentile(trunc_results,[2.5,97.5])
        i+= N_chains



colours = {p_beta_move: (0,i/(len(p_beta_move_values)-1),1 - (i/(len(p_beta_move_values)-1))) for i,p_beta_move in enumerate(p_beta_move_values) }

fig,axs = plt.subplots(5,2,figsize = (50,20))
for dataset_i in range(N_datasets):
    ax = axs[dataset_i%5,dataset_i//5]
    for p_beta_move in p_beta_move_values:
        ax.hist(results_beta_dict[p_beta_move][dataset_i],bins=np.linspace(0.4,0.6,200),alpha = 1,color = colours[p_beta_move],label = p_beta_move,density = True)
axs[0,0].legend()

fn = "figures/p_beta_move_test_plot.png"
if isdir("figures"):
    plt.savefig(fn,dpi=500)
else:
    mkdir("figures")
    plt.savefig(fn,dpi=500)
print(f"Figure saved to {fn}")

fig2, axs2 = plt.subplots(N_datasets, 1, figsize=(12, 3 * N_datasets), sharex=True)
if N_datasets == 1:
    axs2 = [axs2]
for dataset_i in range(N_datasets):
    for p_beta_move in p_beta_move_values:
        ci = results_beta_95CI_dict[p_beta_move][dataset_i]
        x = np.arange(1, ci.shape[0] + 1) * 10 + 100
        axs2[dataset_i].plot(x, ci[:, 0], color=colours[p_beta_move],linestyle='--', label=f'{p_beta_move} 2.5%')
        axs2[dataset_i].plot(x, ci[:, 1], color=colours[p_beta_move], linestyle='--', label=f'{p_beta_move} 97.5%')
    axs2[dataset_i].set_title(f'Dataset {dataset_i}')
    axs2[dataset_i].set_ylabel('Beta 95% CI')
    # Draw a thick hline for the real value (0.5) on each subplot
    ci_final = results_beta_95CI_dict[p_beta_move_values[-1]][dataset_i][-1]
    color = 'green' if ci_final[0] <= 0.5 <= ci_final[1] else 'red'
    xlim = axs2[dataset_i].get_xlim()
    # Draw only a small proportion (e.g., 20%) of the x-axis length, centered
    x_center = (xlim[0] + xlim[1]) / 2
    x_span = (xlim[1] - xlim[0]) * 0.2
    x_start = x_center - x_span / 2
    x_end = x_center + x_span / 2
    axs2[dataset_i].plot([x_start, x_end], [0.5, 0.5], color=color, linewidth=4, solid_capstyle='round')
axs2[-1].set_xlabel('Particle Index')
axs2[0].legend()
plt.tight_layout()
fig2_fn = "figures/p_beta_move_test_95CI_plot.png"
plt.savefig(fig2_fn, dpi=300)
print(f"95% CI figure saved to {fig2_fn}")




