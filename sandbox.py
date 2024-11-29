import os
os.chdir("C:\\Users\\j83508jb\\GitHub Repos\\hh-partition-mcmc")

from HH_case_partition_MCMC import *
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

m = 5 #max hh size
max_k = int(0.5*m*(m+3))

dot_for_cases = np.concatenate([np.arange(0,n+1) for n in range(1,m+1)])
dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])

C0 = np.arange(max_k,0,-1)

n_iters = 100000

C,llhs = RunPartitionsMCMC(C0, 1.5, m, n_iters)