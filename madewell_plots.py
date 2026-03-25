#region Imports
from pickle import load, dump
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile
import pandas as pd

plt.style.use("ggplot")
plt.rcParams["font.family"] = "monospace"
#endregion

#region load datasets
filepath = 'datasets/household_studies_low_info_madewell.csv'

if isfile(filepath):
    df = pd.read_csv(filepath)
else:
    print(f"File {filepath} not found.")
    quit()
    
dataset_names = np.array(df["First Author"]).astype(str).tolist()
N_households = np.array(df["Number of Households"]).astype(int)
mean_sizes = 1 + np.array(df["Number of household contacts"]).astype(float) / np.array(df["Number of Households"]).astype(float)
mean_sizes = [round(m,3) for m in mean_sizes]
SAR_values = np.array(df["Number of household secondary cases"]).astype(float) / np.array(df["Number of household contacts"]).astype(float)

variant_values = np.array(df["Variant"]).astype(str).tolist()
color_dict = {"Pre-Alpha": "blue", "Alpha": "green", "Delta": "orange", "Omicron": "red","Not Specified/Other": "black"}
variant_values = [v if v in color_dict.keys() else "Not Specified/Other" for v in variant_values]

fixed = True 
eta_fixed = 1.

S  = 100

infectious_period_assumption_dict = {"Fixed": lambda t: np.exp(-t),
                                     "Markov": lambda t: 1/(1+t),
                                     "Gamma2": lambda t: (1+t/2)**(-2)}
inf_period_str = "Gamma2" 
phi = infectious_period_assumption_dict[inf_period_str]




results_fn = "outputs/madewell_fits_results"
if fixed:
    results_suffix = f"_eta={eta_fixed}_{inf_period_str}_S={S}"
else:
    results_suffix = f"_Carazo_eta_{inf_period_str}_S={S}"



results_fn = results_fn + results_suffix + f".pkl"


if isfile(results_fn):
    with open(results_fn, 'rb') as f:
        results = load(f)
else:
    print(f"File {results_fn} not found.")
    quit()

index_arr = results[0]
results_arr = results[1]
beta_results = {}
beta_means = {}
beta_CI = {}
n_contacts = 2  #Fixed number of contacts for SITP estimates
SITP_results = {}
SITP_means = {} 
SITP_CI = {}
eta_results = {}
C_results = {}

N_datasets = max(index_arr[:,1]) + 1



for i in range(N_datasets):
    ds_beta_results = []
    ds_eta_results = []
    ds_C_results = []
    for j in range(len(index_arr)):
        if index_arr[j,1] == i:
            beta_chain = results_arr[j][2]
            burn_in = len(beta_chain)//5
            beta_chain = beta_chain[burn_in:]
            ds_beta_results.append(beta_chain)
            
            eta_chain = results_arr[j][3][burn_in:]
            ds_eta_results.append(eta_chain)
            
            C_chain = results_arr[j][0][burn_in:]
            ds_C_results.append(C_chain)
            
            
    beta_results[i] = np.concatenate(ds_beta_results)
    beta_means[i] = np.mean(beta_results[i])
    beta_CI[i] = (np.percentile(beta_results[i], 2.5), np.percentile(beta_results[i], 97.5))
    
    
    
    eta_results[i] = np.concatenate(ds_eta_results)
    
    SITP_results[i] = 1-np.exp(-beta_results[i]/(n_contacts**eta_results[i]))
    SITP_means[i] = np.mean(SITP_results[i])
    SITP_CI[i] = (np.percentile(SITP_results[i], 2.5), np.percentile(SITP_results[i], 97.5))
    
    C_results[i] = np.concatenate(ds_C_results)
#endregion

#region PLOT: \beta trace plots
n_datasets_per_plot = 10
n_plots = N_datasets//n_datasets_per_plot + (1 if N_datasets%n_datasets_per_plot > 0 else 0)
for k in range(n_plots): 
    if k == n_plots-1:
        n_rows = (N_datasets - n_datasets_per_plot*k + 1)//2
    else:
        n_rows = n_datasets_per_plot//2 
    fig,axs = plt.subplots(n_rows,2, figsize=(20,3.5*n_rows))
    for i in range(n_rows*2):
        dataset_i = i+k*n_datasets_per_plot
        ax = axs[i//2,i%2]
        if i%2 ==0:
            ax.set_ylabel("Transmission Rate" +  r"($\beta$)", fontsize=18)
        ax.plot(beta_results[dataset_i],color = color_dict[str(variant_values[dataset_i])],linewidth = 0.25)
        ax.set_title(dataset_names[dataset_i],fontsize=18)
    plt.tight_layout()
    plt.savefig(f"figures/madewell_beta_chains_{results_suffix}_{k}.png", bbox_inches='tight')
print("Beta trace plots saved.")

beta_CI_arr = np.array([beta_CI[i] for i in range(N_datasets)])
beta_x_max = max(beta_CI_arr[:,1])
#endregion

fig,axs = plt.subplots(1,1, figsize=(15,20),sharey = True)

# Sort datasets by SITP means, grouped by variant
to_concat = []
for k in color_dict.keys():
    L = [i  for i,v in enumerate(variant_values) if v==k]
    L_sorted = sorted(L,key=lambda i: SITP_means[i])
    to_concat.append(L_sorted)

sorted_indices = np.concatenate(to_concat)

for plot_pos, i in enumerate(sorted_indices):
    if variant_values[i] in color_dict:
        color = color_dict[str(variant_values[i])]
    else:
        color = "grey"
    axs.errorbar(SITP_means[i], plot_pos, xerr=[[SITP_means[i] - SITP_CI[i][0]], [SITP_CI[i][1] - SITP_means[i]]], fmt='o', capsize=10, markersize=10,color = color,elinewidth = 3)
    
sorted_labels = [dataset_names[i] for i in sorted_indices]

axs.set_xlim(0,1)

axs.set_xlabel(f'Estimated SITP (Household of {n_contacts})', fontsize=30)
axs.set_yticks(range(N_datasets), labels=sorted_labels, rotation=0, ha='right',fontsize=15)


plt.savefig(f'figures/madewell_SITP_estimates_n={n_contacts}{results_suffix}.png', bbox_inches='tight')

fig,axs = plt.subplots(1,1, figsize=(15,20),sharey = True)

to_concat = []
for k in color_dict.keys():
    L = [i  for i,v in enumerate(variant_values) if v==k]
    L_sorted = sorted(L,key=lambda i: beta_means[i])
    to_concat.append(L_sorted)

sorted_indices = np.concatenate(to_concat)

current_variant = list(color_dict.keys())[0]
axs.text(x=beta_x_max,y=0,s=current_variant,fontsize = 40,horizontalalignment="right",color = color_dict[current_variant])#,name="Courier")
buffer = 0
tick_pos = []
for plot_pos, i in enumerate(sorted_indices):
    variant = variant_values[i]
    color = color_dict[variant]
    if variant!=current_variant:
        buffer +=1
        current_variant = variant
        axs.hlines([plot_pos+buffer-1],0,100,color = "grey", linewidth=2, linestyle = "--")
        axs.text(x=beta_x_max,y=plot_pos+buffer,s=current_variant,fontsize = 40,horizontalalignment="right",color = color)#,name="Courier")
    

    tick_pos.append(plot_pos+buffer)
    axs.errorbar(beta_means[i], plot_pos+buffer, xerr=[[beta_means[i] -beta_CI[i][0]], [beta_CI[i][1] - beta_means[i]]], fmt='o', capsize=10, markersize=10,color = color,elinewidth = 3)

sorted_labels = [dataset_names[i] for i in sorted_indices]
axs.set_xlim(-0. ,beta_x_max*1.05)

axs.set_xlabel(r'Estimated Transmission Rate ($\beta$)', fontsize=30)
axs.set_yticks(tick_pos, labels=sorted_labels, rotation=0, ha='right',fontsize=15)
axs.set_xticks(np.arange(0,beta_x_max*1.051,0.25))
axs.set_ylim(-0.5,plot_pos+buffer+1)


plt.savefig(f'figures/madewell_beta_estimates{results_suffix}.png', bbox_inches='tight')


data_for_table = {"SAR": [round(SAR_values[i],3) for i in sorted_indices[::-1]], # type: ignore
                  "Mean Size": [mean_sizes[i] for i in sorted_indices[::-1]],
                  "N": [N_households[i] for i in sorted_indices[::-1]],}
df_for_table = pd.DataFrame(data_for_table)

summary = [(f"{dataset_names[i]}, {beta_means[i]}, {beta_CI[i]}") for i in sorted_indices[::-1]]
for line in summary:
    print(line)

print(df_for_table.to_latex(index=False, bold_rows=True,float_format="%.3f"))

