import time
import math
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import matplotlib.pyplot as plt

# set timer
start_time = time.time()

# set up feature & label sizes 
input_layer_size=230400;  # 480x480 Input Images of Digits
hidden_layer_size=40;     # 40 hidden units
num_labels=3;             # 3 labels: exponential, surface, interior  

r_train=np.array([float(line) for line in open('../data/radius_train.dat','r').readlines()])
r_cross=np.array([float(line) for line in open('../data/radius_cross.dat','r').readlines()])
r_test=np.array([float(line) for line in open('../data/radius_test.dat','r').readlines()])
phi_train=np.array([float(line) for line in open('../data/phi_train.dat','r').readlines()])
phi_cross=np.array([float(line) for line in open('../data/phi_cross.dat','r').readlines()])
phi_test=np.array([float(line) for line in open('../data/phi_test.dat','r').readlines()])
y_train=np.array([float(line) for line in open('../data/y_train.dat','r').readlines()])
y_cross=np.array([float(line) for line in open('../data/y_cross.dat','r').readlines()])
y_test=np.array([float(line) for line in open('../data/y_test.dat','r').readlines()])

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.plot(r_train[0],phi_train[0],'or',markersize=4.5,label='Axial')
plt.plot(r_train[0],phi_train[0],'og',markersize=4.5,label='Polar 1')        
plt.plot(r_train[0],phi_train[0],'ob',markersize=4.5,label='Random')
for i in range(1,len(r_train)):
    if(y_train[i]==1):
        plt.plot(r_train[i],phi_train[i],'or',markersize=4.5)
    elif(y_train[i]==2):
        plt.plot(r_train[i],phi_train[i],'og',markersize=4.5)      
    else:
        plt.plot(r_train[i],phi_train[i],'ob',markersize=4.5)
plt.yticks([0.93,0.935,0.94,0.945,0.95])
plt.xlabel('Radius, $R$',fontsize=16)
plt.ylabel('Packing fraction, $\phi$',fontsize=16)
plt.legend(frameon=False,handletextpad=0.08,loc='lower right',fontsize=13)
plt.xlim([13,17.5])
plt.ylim([0.93,0.95])
plt.tight_layout()
plt.savefig('../figures/R_phi_scatter_train.png')
plt.show()

# make reduced feature vectors
X_train_red=np.transpose(np.array([r_train,phi_train]))
X_cross_red=np.transpose(np.array([r_cross,phi_cross]))
X_test_red=np.transpose(np.array([r_test,phi_test]))

# normalize data
Xmean=np.mean(X_train_red,axis=0)
Xstd=np.std(X_train_red,axis=0)
X_train_red=(X_train_red-Xmean)/Xstd
X_cross_red=(X_cross_red-Xmean)/Xstd
X_test_red=(X_test_red-Xmean)/Xstd

# reduce training data size
m=5040;
y_train=y_train[0:m];
X_train_red=X_train_red[0:m,:];

num_iter=6000

clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainAA=sum(p_train==y_train)/len(y_train)
acrossAA=sum(p_cross==y_cross)/len(y_cross)
atestAA=sum(p_test==y_test)/len(y_test)
print(atrainAA)
print(acrossAA)
print(atestAA)
print()
dump(clf, 'fit_params_nn_simple_A.joblib')

clf=MLPClassifier(solver='lbfgs',alpha=1e-3,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainA=sum(p_train==y_train)/len(y_train)
acrossA=sum(p_cross==y_cross)/len(y_cross)
atestA=sum(p_test==y_test)/len(y_test)
print(atrainA)
print(acrossA)
print(atestA)
print()
dump(clf, 'fit_params_nn_simple_A.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=2e-3,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainB=sum(p_train==y_train)/len(y_train)
acrossB=sum(p_cross==y_cross)/len(y_cross)
atestB=sum(p_test==y_test)/len(y_test)
print(atrainB)
print(acrossB)
print(atestB)
print()
dump(clf, 'fit_params_nn_simple_B.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=5e-3,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainC=sum(p_train==y_train)/len(y_train)
acrossC=sum(p_cross==y_cross)/len(y_cross)
atestC=sum(p_test==y_test)/len(y_test)
print(atrainC)
print(acrossC)
print(atestC)
print()
dump(clf, 'fit_params_nn_simple_C.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=1e-2,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainD=sum(p_train==y_train)/len(y_train)
acrossD=sum(p_cross==y_cross)/len(y_cross)
atestD=sum(p_test==y_test)/len(y_test)
print(atrainD)
print(acrossD)
print(atestD)
print()
dump(clf, 'fit_params_nn_simple_D.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=2e-2,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainE=sum(p_train==y_train)/len(y_train)
acrossE=sum(p_cross==y_cross)/len(y_cross)
atestE=sum(p_test==y_test)/len(y_test)
print(atrainE)
print(acrossE)
print(atestE)
print()
dump(clf, 'fit_params_nn_simple_E.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=5e-2,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainF=sum(p_train==y_train)/len(y_train)
acrossF=sum(p_cross==y_cross)/len(y_cross)
atestF=sum(p_test==y_test)/len(y_test)
print(atrainF)
print(acrossF)
print(atestF)
print()
dump(clf, 'fit_params_nn_simple_F.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=1e-1,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainG=sum(p_train==y_train)/len(y_train)
acrossG=sum(p_cross==y_cross)/len(y_cross)
atestG=sum(p_test==y_test)/len(y_test)
print(atrainG)
print(acrossG)
print(atestG)
print()
dump(clf, 'fit_params_nn_simple_G.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=2e-1,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainH=sum(p_train==y_train)/len(y_train)
acrossH=sum(p_cross==y_cross)/len(y_cross)
atestH=sum(p_test==y_test)/len(y_test)
print(atrainH)
print(acrossH)
print(atestH)
print()
dump(clf, 'fit_params_nn_simple_H.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=5e-1,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainI=sum(p_train==y_train)/len(y_train)
acrossI=sum(p_cross==y_cross)/len(y_cross)
atestI=sum(p_test==y_test)/len(y_test)
print(atrainI)
print(acrossI)
print(atestI)
print()
dump(clf, 'fit_params_nn_simple_I.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=1e0,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainJ=sum(p_train==y_train)/len(y_train)
acrossJ=sum(p_cross==y_cross)/len(y_cross)
atestJ=sum(p_test==y_test)/len(y_test)
print(atrainJ)
print(acrossJ)
print(atestJ)
print()
dump(clf, 'fit_params_nn_simple_J.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=2e0,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
atrainK=sum(p_train==y_train)/len(y_train)
acrossK=sum(p_cross==y_cross)/len(y_cross)
atestK=sum(p_test==y_test)/len(y_test)
print(atrainK)
print(acrossK)
print(atestK)
print()
dump(clf, 'fit_params_nn_simple_K.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=5e0,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainL=sum(p_train==y_train)/len(y_train)
acrossL=sum(p_cross==y_cross)/len(y_cross)
atestL=sum(p_test==y_test)/len(y_test)
print(atrainL)
print(acrossL)
print(atestL)
print()
dump(clf, 'fit_params_nn_simple_L.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=1e1,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainM=sum(p_train==y_train)/len(y_train)
acrossM=sum(p_cross==y_cross)/len(y_cross)
atestM=sum(p_test==y_test)/len(y_test)
print(atrainM)
print(acrossM)
print(atestM)
print()
dump(clf, 'fit_params_nn_simple_M.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=2e1,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainN=sum(p_train==y_train)/len(y_train)
acrossN=sum(p_cross==y_cross)/len(y_cross)
atestN=sum(p_test==y_test)/len(y_test)
print(atrainN)
print(acrossN)
print(atestN)
print()
dump(clf, 'fit_params_nn_simple_N.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=5e1,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainO=sum(p_train==y_train)/len(y_train)
acrossO=sum(p_cross==y_cross)/len(y_cross)
atestO=sum(p_test==y_test)/len(y_test)
print(atrainO)
print(acrossO)
print(atestO)
print()
dump(clf, 'fit_params_nn_simple_O.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=1e2,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainP=sum(p_train==y_train)/len(y_train)
acrossP=sum(p_cross==y_cross)/len(y_cross)
atestP=sum(p_test==y_test)/len(y_test)
print(atrainP)
print(acrossP)
print(atestP)
print()
dump(clf, 'fit_params_nn_simple_P.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=2e2,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainQ=sum(p_train==y_train)/len(y_train)
acrossQ=sum(p_cross==y_cross)/len(y_cross)
atestQ=sum(p_test==y_test)/len(y_test)
print(atrainQ)
print(acrossQ)
print(atestQ)
print()
dump(clf, 'fit_params_nn_simple_Q.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=5e2,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainR=sum(p_train==y_train)/len(y_train)
acrossR=sum(p_cross==y_cross)/len(y_cross)
atestR=sum(p_test==y_test)/len(y_test)
print(atrainR)
print(acrossR)
print(atestR)
print()
dump(clf, 'fit_params_nn_simple_R.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=7e2,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainS=sum(p_train==y_train)/len(y_train)
acrossS=sum(p_cross==y_cross)/len(y_cross)
atestS=sum(p_test==y_test)/len(y_test)
print(atrainS)
print(acrossS)
print(atestS)
print()
dump(clf, 'fit_params_nn_simple_S.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=1e3,max_iter=num_iter,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
atrainT=sum(p_train==y_train)/len(y_train)
acrossT=sum(p_cross==y_cross)/len(y_cross)
atestT=sum(p_test==y_test)/len(y_test)
print(atrainT)
print(acrossT)
print(atestT)
print()
dump(clf, 'fit_params_nn_simple_T.joblib')

regparams=np.array([1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1e0,2e0,5e0,1e1,2e1,5e1,1e2,2e2,5e2,7e2,1e3])
atrain=np.array([atrainA,atrainB,atrainC,atrainD,atrainE,atrainF,atrainG,atrainH,atrainI,atrainJ,atrainK,atrainL,atrainM,atrainN,atrainO,atrainP,atrainQ,atrainR,atrainS,atrainT]);
across=np.array([acrossA,acrossB,acrossC,acrossD,acrossE,acrossF,acrossG,acrossH,acrossI,acrossJ,acrossK,acrossL,acrossM,acrossN,acrossO,acrossP,acrossQ,acrossR,acrossS,acrossT]);
atest=np.array([atestA,atestB,atestC,atestD,atestE,atestF,atestG,atestH,atestI,atestJ,atestK,atestL,atestM,atestN,atestO,atestP,atestQ,atestR,atestS,atestT]);

plt.xscale('log')
plt.plot(regparams,atrain,'-ok',regparams,across,'-sr')
plt.show()

np.savetxt('../data/acc_R_phi_train.dat',np.c_[regparams,atrain])
np.savetxt('../data/acc_R_phi_cross.dat',np.c_[regparams,across])
np.savetxt('../data/acc_R_phi_test.dat',np.c_[regparams,atest])

# print running time
print("Running time:",int(time.time()-start_time),"seconds")
