from pickle import load, dump
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from os.path import isfile, isdir 
from os import mkdir
from tqdm import tqdm
from HH_case_partition_MCMC import IndexChange1dTo2d

plt.style.use('ggplot')

with open("datasets/synthetic.pkl", "rb") as f:
    synthetic_datasets = load(f)

beta_real = 0.5
eta_real = 0.0
N_hh=1000

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

partition_results = array([r[0] for r in results])
beta_results = array([r[1] for r in results])

# plt.plot(beta_results[0][:],label = params[0])
# plt.plot(beta_results[-1],label = params[-1])
# plt.legend()
# plt.show()
# exit()

p_beta_move_values = np.unique(params[:,0])
N_datasets = int(max(params[:,1]))+1
N_chains = int(max(params[:,2]))+1
N_particles_per_chain, k_max = partition_results.shape[1:]

partition_results_dict = {p_beta_move: np.zeros((N_datasets,N_chains*(N_particles_per_chain-N_particles_per_chain//5),k_max))
                                         for p_beta_move in p_beta_move_values}

i=0
for p_beta_move in p_beta_move_values:
    for dataset_i in range(N_datasets):
        partition_results_dict[p_beta_move][dataset_i] = np.concatenate(partition_results[i:i+N_chains,N_particles_per_chain//5:])
        i+=N_chains

partition_CI_dict = {p_beta_move: np.zeros((N_datasets,2,k_max))
                                    for p_beta_move in p_beta_move_values}

for p_beta_move in p_beta_move_values:
    for dataset_i in range(N_datasets):
        partitions = partition_results_dict[p_beta_move][dataset_i]
        partition_CI_dict[p_beta_move][dataset_i] = np.percentile(partitions, [2.5, 97.5], axis=0)

# --- Add boxplot subplots for each dataset, arranged in 2 rows ---

ticklabels1 = [r"$\mathbf{(n,y)}$"]
ticklabels2 = [IndexChange1dTo2d(i) for i in range(k_max)]
ticklabels2 = [f"({n},{y})" for n,y in ticklabels2]
ticklabels3 = ticklabels1 + ticklabels2
n_rows = 4
n_cols = int(np.ceil(N_datasets / n_rows))
for p_beta_move in p_beta_move_values:
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 6*n_rows), squeeze=False)
    fig.suptitle(f'Partition Boxplots for p_beta_move={p_beta_move}', fontsize=20)
    for dataset_i in range(N_datasets):
        row = dataset_i // n_cols
        col = dataset_i % n_cols
        ax = axs[row, col]
        ax.boxplot(partition_results_dict[p_beta_move][dataset_i])
        ax.plot(np.arange(1,k_max+1,1),synthetic_datasets[beta_real][eta_real][N_hh]["UK"][dataset_i],label = "Synthetic dataset")
        ax.set_title(f'Dataset {dataset_i}')
        if row == n_rows-1:
            ax.set_xlabel('Partition Index')
        if col == 0:
            ax.set_ylabel('Count')
        ax.set_xticks(np.arange(0,k_max+1,1))
        ax.set_xticklabels(ticklabels3, rotation=45, ha='center')
        ax.set_xlim(0.5, k_max+0.5)
    # Hide unused subplots if any
    for idx in range(N_datasets, n_rows*n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axs[row, col])
    plt.tight_layout()
    plt.savefig(f"figures\\p_beta_move_test_p={p_beta_move}_plot.png")



#         N_particles_per_chain = len(results[0][2])

#         partition_results = [array([c for c in r[0]]) for r in tqdm(results)] #Dont think the IDs are worth using
#         k_max = len(partition_results[0][0])
#         partition_results_dict = {beta:{eta:{N_hh: np.zeros((N_datasets,N_chains*(N_particles_per_chain-N_particles_per_chain//5),k_max))
#                                         for N_hh in N_hh_values}
#                             for eta in eta_values}
#                         for beta in beta_values}

#         i = 0
#         for beta in beta_values:
#             for eta in eta_values:
#                 for N_hh in N_hh_values:
#                     for dataset_i in range(N_datasets):
#                         partition_results_dict[beta][eta][N_hh][dataset_i] =  np.concatenate([partition_results[i+j][N_particles_per_chain//5:] for j in range(N_chains)])
#                         i+= N_chains

#         with open(results_fn[:-4] + "_partition_dict.pkl", "wb") as f:
#             dump(partition_results_dict, f)
#     else:
#         print("Results dictionary found, loading...")
#         with open(results_fn[:-4] + "_partition_dict.pkl", "rb") as f:
#             partition_results_dict = load(f)
        
#         beta_values = sorted(partition_results_dict.keys())
#         eta_values = sorted(partition_results_dict[beta_values[0]].keys())
#         N_hh_values = sorted(partition_results_dict[beta_values[0]][eta_values[0]].keys())
#         N_datasets = len(partition_results_dict[beta_values[0]][eta_values[0]][N_hh_values[0]])

#     k_max = 20
#     partition_CI_dict = {beta:{eta:{N_hh: np.zeros((N_datasets,2,k_max))
#                                     for N_hh in N_hh_values}
#                         for eta in eta_values}
#                     for beta in beta_values}
#     for beta in beta_values:
#         for eta in eta_values:
#             for N_hh in N_hh_values:
#                 for dataset_i in range(N_datasets):
#                     partitions = partition_results_dict[beta][eta][N_hh][dataset_i]
#                     partition_CI_dict[beta][eta][N_hh][dataset_i] = np.percentile(partitions, [2.5, 97.5], axis=0)



#     beta = 0.5

#     ticklabels1 = [r"$\mathbf{(n,y)}$"]
#     ticklabels2 = [IndexChange1dTo2d(i) for i in range(k_max)]
#     ticklabels2 = [f"({n},{y})" for n,y in ticklabels2]
#     ticklabels3 = ticklabels1 + ticklabels2
    
# for dataset_i in range(10):
#     fig, axs = plt.subplots(len(eta_values), len(N_hh_values), figsize=(20, 15))
#     fig.supylabel("Number of households",fontsize=20)
#     for i, eta in enumerate(eta_values):
#         for j, N_hh in enumerate(N_hh_values):
#             ax = axs[i, j]
#             ax = axs[i, j]
#             ax.set_xlim(0, 1)
#             ax.set_title(f'eta={eta}, N_hh={N_hh}')
#             ax.set_xlim(-0.5,k_max+0.5)
            
#             ax.boxplot(x=partition_results_dict[beta][eta][N_hh][dataset_i])
#             ax.plot(np.arange(1,k_max+1,1),synthetic_datasets[beta][eta][N_hh][size_dist_key][dataset_i])
#             ax.scatter(np.arange(1,k_max+1,1),synthetic_datasets[beta][eta][N_hh][size_dist_key][dataset_i],marker="x")
#             ax.set_xticks(np.arange(0,k_max+1,1))
#             ax.set_xticklabels(ticklabels3, rotation=45, ha='center')
#     fn = "figures/partition plots/" + results_fn[8:-4] + "_partitions_plot_beta=" + str(beta) + "_" + str(dataset_i) + ".png"
#     plt.tight_layout()
#     plt.savefig(fn)

