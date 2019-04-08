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

# load pickle data - training set split into 3
data_pickle=pickle.load(open("image_data_train_A.pkl","rb"))
X_A=data_pickle[0].astype(float)
y_A=data_pickle[1]
data_pickle=pickle.load(open("image_data_train_B.pkl","rb"))
X_B=data_pickle[0].astype(float)
y_B=data_pickle[1]
data_pickle=pickle.load(open("image_data_train_C.pkl","rb"))
X_C=data_pickle[0].astype(float)
y_C=data_pickle[1]
data_pickle=pickle.load(open("image_data_train_D.pkl","rb"))
X_D=data_pickle[0].astype(float)
y_D=data_pickle[1]
data_pickle=pickle.load(open("image_data_train_E.pkl","rb"))
X_E=data_pickle[0].astype(float)
y_E=data_pickle[1]
data_pickle=pickle.load(open("image_data_train_F.pkl","rb"))
X_F=data_pickle[0].astype(float)
y_F=data_pickle[1]
X_train=np.concatenate((X_A,X_B,X_C,X_D,X_E,X_F),axis=0)
y_train=np.concatenate((y_A,y_B,y_C,y_D,y_E,y_F),axis=0)
del(X_A,X_B,X_C,X_D,X_E,X_F,y_A,y_B,y_C,y_D,y_E,y_F)

data_pickle=pickle.load(open("image_data_cross.pkl","rb"))
X_cross=data_pickle[0].astype(float)
y_cross=data_pickle[1]
data_pickle=pickle.load(open("image_data_test.pkl","rb"))
X_test=data_pickle[0].astype(float)
y_test=data_pickle[1]
del(data_pickle)

# Convert to binary: 1 for pixels containg cells, 0 otherwise
X_train=np.round(1.-(X_train-128.)/127.);
X_cross=np.round(1.-(X_cross-128.)/127.);
X_test=np.round(1.-(X_test-128.)/127.);

ncols=np.arange(0,len(X_cross[0,:]))%480
nrows=np.transpose(ncols.reshape(480,480)).reshape(ncols.shape)

a_train=np.zeros(len(y_train))
r_train=np.zeros(len(y_train))
phi_train=np.zeros(len(y_train))
scalesq=(40./480.)**2
for i in range(0,len(X_train)):
    pix=X_train[i,:]
    tot=np.sum(pix)
    xmean=np.sum(np.multiply(ncols,pix))/tot
    ymean=np.sum(np.multiply(nrows,pix))/tot
    Rgsq=scalesq*np.sum(np.multiply(np.power(ncols-xmean,2)+np.power(nrows-ymean,2),pix))/tot
    a_train[i]=tot*scalesq    
    r_train[i]=math.sqrt(2.*Rgsq)
    phi_train[i]=a_train[i]/(math.pi*pow(r_train[i],2))

a_cross=np.zeros(len(y_cross))
r_cross=np.zeros(len(y_cross))
phi_cross=np.zeros(len(y_cross))
scalesq=(40./480.)**2
for i in range(0,len(X_cross)):
    pix=X_cross[i,:]
    tot=np.sum(pix)
    xmean=np.sum(np.multiply(ncols,pix))/tot
    ymean=np.sum(np.multiply(nrows,pix))/tot
    Rgsq=scalesq*np.sum(np.multiply(np.power(ncols-xmean,2)+np.power(nrows-ymean,2),pix))/tot
    a_cross[i]=tot*scalesq    
    r_cross[i]=math.sqrt(2.*Rgsq)
    phi_cross[i]=a_cross[i]/(math.pi*pow(r_cross[i],2))

a_test=np.zeros(len(y_test))
r_test=np.zeros(len(y_test))
phi_test=np.zeros(len(y_test))
scalesq=(40./480.)**2
for i in range(0,len(X_test)):
    pix=X_test[i,:]
    tot=np.sum(pix)
    xmean=np.sum(np.multiply(ncols,pix))/tot
    ymean=np.sum(np.multiply(nrows,pix))/tot
    Rgsq=scalesq*np.sum(np.multiply(np.power(ncols-xmean,2)+np.power(nrows-ymean,2),pix))/tot
    a_test[i]=tot*scalesq    
    r_test[i]=math.sqrt(2.*Rgsq)
    phi_test[i]=a_test[i]/(math.pi*pow(r_test[i],2))

np.savetxt('../data/radius_train.dat',r_train,delimiter=',',fmt='%1.5e') 
np.savetxt('../data/radius_cross.dat',r_cross,delimiter=',',fmt='%1.5e') 
np.savetxt('../data/radius_test.dat',r_test,delimiter=',',fmt='%1.5e') 
np.savetxt('../data/phi_train.dat',phi_train,delimiter=',',fmt='%1.5e')
np.savetxt('../data/phi_cross.dat',phi_cross,delimiter=',',fmt='%1.5e')
np.savetxt('../data/phi_test.dat',phi_test,delimiter=',',fmt='%1.5e')
np.savetxt('../data/y_train.dat',y_train,delimiter=',',fmt='%1.5e')
np.savetxt('../data/y_cross.dat',y_cross,delimiter=',',fmt='%1.5e')
np.savetxt('../data/y_test.dat',y_test,delimiter=',',fmt='%1.5e')
