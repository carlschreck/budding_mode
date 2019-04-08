import time
import math
import pickle
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
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

num_iter=1000 #6000

# train random forest
clf=RandomForestClassifier(n_estimators=500,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=50,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')
perfmat=np.zeros((3,3))
for i in range(0,len(p_test)):
    if(y_test[i]==1 and p_test[i]==1):
        perfmat[0,0]+=1
    elif(y_test[i]==1 and p_test[i]==2):
        perfmat[0,1]+=1
    elif(y_test[i]==1 and p_test[i]==3):
        perfmat[0,2]+=1
    elif(y_test[i]==2 and p_test[i]==1):
        perfmat[1,0]+=1
    elif(y_test[i]==2 and p_test[i]==2):
        perfmat[1,1]+=1
    elif(y_test[i]==2 and p_test[i]==3):
        perfmat[1,2]+=1
    elif(y_test[i]==3 and p_test[i]==1):
        perfmat[2,0]+=1
    elif(y_test[i]==3 and p_test[i]==2):
        perfmat[2,1]+=1
    elif(y_test[i]==3 and p_test[i]==3):
        perfmat[2,2]+=1
print(perfmat)
print()

# train random forest
clf=RandomForestClassifier(n_estimators=48,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=46,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=42,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=40,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=38,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=36,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=34,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=32,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=30,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=28,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=26,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=24,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=22,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=20,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=18,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=16,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=14,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=12,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=10,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib') 

# train random forest
clf=RandomForestClassifier(n_estimators=8,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=6,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=5,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=4,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=3,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=2,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

# train random forest
clf=RandomForestClassifier(n_estimators=1,random_state=0)
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_svm_test.joblib')

#regparams=np.array([1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1e0,2e0,5e0,1e1,2e1,5e1,1e2,2e2,5e2,7e2,1e3])
#atrain=np.array([atrainA,atrainB,atrainC,atrainD,atrainE,atrainF,atrainG,atrainH,atrainI,atrainJ,atrainK,atrainL,atrainM,atrainN,atrainO,atrainP,atrainQ,atrainR,atrainS,atrainT]);
#across=np.array([acrossA,acrossB,acrossC,acrossD,acrossE,acrossF,acrossG,acrossH,acrossI,acrossJ,acrossK,acrossL,acrossM,acrossN,acrossO,acrossP,acrossQ,acrossR,acrossS,acrossT]);
#atest=np.array([atestA,atestB,atestC,atestD,atestE,atestF,atestG,atestH,atestI,atestJ,atestK,atestL,atestM,atestN,atestO,atestP,atestQ,atestR,atestS,atestT]);
#
#plt.xscale('log')
#plt.plot(regparams,atrain,'-ok',regparams,across,'-sr')
#plt.show()

#np.savetxt('../data/acc_R_phi_train.dat',np.c_[regparams,atrain])
#np.savetxt('../data/acc_R_phi_cross.dat',np.c_[regparams,across])
#np.savetxt('../data/acc_R_phi_test.dat',np.c_[regparams,atest])

# print running time
print("Running time:",int(time.time()-start_time),"seconds")
