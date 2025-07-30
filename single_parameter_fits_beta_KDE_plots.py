from pickle import load, dump
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from os.path import isfile, isdir
from os import mkdir
from tqdm import tqdm

plt.style.use('ggplot')

if not (isfile("outputs/single_parameter_fit_params.pkl") and isfile("outputs/single_parameter_fit_results.pkl")):
    raise FileNotFoundError("Required files not found. Please run the fitting script first.")





else:
    if not isfile("outputs/single_parameter_fit_beta_results_dict.pkl"):
        print("Results dictionary not found, creating...")


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

        beta_results_dict = {beta:{eta:{N_hh: np.zeros((N_datasets,N_chains*(N_particles_per_chain - N_particles_per_chain//5)))
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

        with open("outputs/single_parameter_fit_beta_results_dict.pkl", "wb") as f:
            dump(beta_results_dict, f)
    else:
        print("Results dictionary found, loading...")
        with open("outputs/single_parameter_fit_beta_results_dict.pkl", "rb") as f:
            beta_results_dict = load(f)

    beta_values = sorted(beta_results_dict.keys())
    eta_values = sorted(beta_results_dict[beta_values[0]].keys())
    N_hh_values = sorted(beta_results_dict[beta_values[0]][eta_values[0]].keys())
    N_datasets = len(beta_results_dict[beta_values[0]][eta_values[0]][N_hh_values[0]])

    if not isfile("outputs/single_parameter_fit_beta_results_KDE_dict.pkl"):
        print("KDEs not found, calculating...")
        KDE_dict = {beta:{eta:{N_hh: [gaussian_kde(beta_results_dict[beta][eta][N_hh][dataset_i]) for dataset_i in range(N_datasets)]
                        for N_hh in N_hh_values}
                        for eta in eta_values}
                        for beta in beta_values}
        with open("outputs/single_parameter_fit_beta_results_KDE_dict.pkl", "wb") as f:
            dump(KDE_dict, f)
    else:
        print("KDEs found, loading...")
        with open("outputs/single_parameter_fit_beta_results_KDE_dict.pkl", "rb") as f:
            KDE_dict = load(f)


    prog_bar = tqdm(total = len(beta_values) * len(eta_values) * len(N_hh_values) * N_datasets, desc="Plotting...")
    colours = {beta: (i/(len(beta_values)-1),1 - (i/(len(beta_values)-1)),0) for i,beta in enumerate(beta_values) }

    X = np.linspace(0,3,300)

    fig, axs = plt.subplots(len(eta_values), len(N_hh_values), figsize=(20, 15),sharex=True)

    for i, eta in enumerate(eta_values):
        for j, N_hh in enumerate(N_hh_values):
            ax = axs[i, j]
            ax.set_xlim(0, 3)
            for beta in beta_values:
                ax.vlines(beta,0,10,color=colours[beta],linestyle="--",label=f'beta={beta}')
                for dataset_i in range(N_datasets):
                    ax.plot(X,KDE_dict[beta][eta][N_hh][dataset_i](X),color = colours[beta],linestyle = "--")
                    prog_bar.update(1)
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
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    fn = "figures/single_parameter_fits_histograms.png"
    if isdir("figures"):
        plt.savefig(fn,dpi=500)
    else:
        mkdir("figures")
        plt.savefig(fn,dpi=500)
    print("Figure saved to ", fn)