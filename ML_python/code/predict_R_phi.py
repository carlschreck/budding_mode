import time
import math
import pickle
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib') 
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


# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=80,random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib') 

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=160,random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=240,random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib') 

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=320,random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=(20,20),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=(40,40),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=(120,120),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=(160,160),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=200,hidden_layer_sizes=(200,200),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=400,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=800,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=1200,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=1600,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=2000,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=2400,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=2800,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=3200,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=3600,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=4000,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=4400,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

# train neural network
clf=MLPClassifier(solver='lbfgs',alpha=0.,max_iter=4800,hidden_layer_sizes=(80,80),random_state=1);
clf.fit(X_train_red,y_train)
p_train=clf.predict(X_train_red);
p_cross=clf.predict(X_cross_red);
p_test=clf.predict(X_test_red);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
#dump(clf, 'fit_params_nn_simple_test.joblib')

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

# print running time
print("Running time:",int(time.time()-start_time),"seconds")
