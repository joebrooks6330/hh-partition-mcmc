import os
os.chdir("C:\\Users\\j83508jb\\GitHub Repos\\hh-partition-mcmc")

from HH_case_partition_MCMC import *

m = 5 #max hh size
max_k = int(0.5*m*(m+3))

dot_for_cases = np.concatenate([np.arange(0,n+1) for n in range(1,m+1)])
dot_for_contacts = np.concatenate([np.zeros(n+1)+n for n in range(1,m+1)])

C0 = np.zeros(max_k)+1

n_iters = 100
u_infected = np.random.uniform(0,1,size=n_iters)
C = np.zeros((n_iters+1,max_k))
C[0] = C0

for i in range(n_iters):
    k1,k2,infected = SelectIndices(C[i], dot_for_contacts, m, u_infected[i])
    C[i+1] = MoveContact(C[i], k1, k2, infected)