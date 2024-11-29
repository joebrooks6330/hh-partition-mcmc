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
total_contacts = C0.dot(dot_for_contacts)

n_iters = 10000

C,llhs = RunPartitionsMCMC(C0, 1.5, m, n_iters)

for i,(c,l) in enumerate(zip(C[1000:],llhs[1000:])):
    contacts = np.zeros(m)
    cases = np.zeros(m)
    for n in range(1,m+1):
        contacts[n-1] = n*sum(c[np.where(dot_for_contacts==n)])
        cases[n-1] = (c[np.where(dot_for_contacts==n)]).dot(np.arange(n+1))
    plt.cla()
    plt.bar(np.arange(2,m+2),contacts,label = "Contacts")
    plt.bar(np.arange(2,m+2),cases,label = "Cases")
    plt.xlabel("Household size")
    plt.ylabel("Count")
    plt.ylim(0,total_contacts)
    plt.text(2,200,l)
    plt.legend()
    plt.savefig("frames//" + str(i))