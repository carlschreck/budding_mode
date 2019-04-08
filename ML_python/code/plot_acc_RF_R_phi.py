import time
import math
import numpy as np
import matplotlib.pyplot as plt

# set timer
start_time = time.time()

regparams=np.array([float(line.split()[0]) for line in open('../data/acc_RF_R_phi_train.dat','r').readlines()])
atrain=np.array([float(line.split()[1]) for line in open('../data/acc_RF_R_phi_train.dat','r').readlines()])
across=np.array([float(line.split()[1]) for line in open('../data/acc_RF_R_phi_cross.dat','r').readlines()])
atest=np.array([float(line.split()[1]) for line in open('../data/acc_RF_R_phi_train.dat','r').readlines()])

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.xscale('log')
plt.plot(regparams,atrain,'-ok',label='Training set')
plt.plot(regparams,across,'--sk',label='Cross-validation set')
plt.plot([1e-1,1e3],[1/3,1/3],':k',label='Baseline, $A_0=1/3$')
plt.plot([1e-1,1e3],[1,1],'-k')
plt.xlabel('Number of trees',fontsize=16)
plt.ylabel('Accuracy, $A$',fontsize=16)
plt.legend(frameon=False,handletextpad=0.15,loc='lower right',fontsize=13)
plt.xlim([5e-1,1e2])
plt.ylim([0.3,1.1])
plt.tight_layout()
plt.savefig('../figures/acc_vs_num_trees_R_phi_RF.png')
plt.show()

# print running time
print("Running time:",int(time.time()-start_time),"seconds")
