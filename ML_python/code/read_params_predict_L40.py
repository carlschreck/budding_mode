import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import matplotlib.pyplot as plt

clf=load('../params/fit_params_nn_image_L40_seed114_alpa1e0_m5040.joblib')
params=clf.coefs_

fig=plt.figure()
for i in range(40):
    img=params[0][:,i].reshape((480,480))
    plt.subplot(5,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,interpolation='nearest')
plt.tight_layout()
fig.set_size_inches(8,5)
plt.show()

# load pickle data - training set split into 3
data_pickle=pickle.load(open("../data/image_data_train_A.pkl","rb"))
X_A=data_pickle[0].astype(float)
y_A=data_pickle[1]
print(1)
data_pickle=pickle.load(open("../data/image_data_train_B.pkl","rb"))
X_B=data_pickle[0].astype(float)
y_B=data_pickle[1]
print(2)
data_pickle=pickle.load(open("../data/image_data_train_C.pkl","rb"))
X_C=data_pickle[0].astype(float)
y_C=data_pickle[1]
print(3)
data_pickle=pickle.load(open("../data/image_data_train_D.pkl","rb"))
X_D=data_pickle[0].astype(float)
y_D=data_pickle[1]
print(4)
data_pickle=pickle.load(open("../data/image_data_train_E.pkl","rb"))
X_E=data_pickle[0].astype(float)
y_E=data_pickle[1]
print(5)
data_pickle=pickle.load(open("../data/image_data_train_F.pkl","rb"))
X_F=data_pickle[0].astype(float)
y_F=data_pickle[1]
print(6)
X_train=np.concatenate((X_A,X_B,X_C,X_D,X_E,X_F),axis=0)
y_train=np.concatenate((y_A,y_B,y_C,y_D,y_E,y_F),axis=0)
del(X_A,X_B,X_C,X_D,X_E,X_F,y_A,y_B,y_C,y_D,y_E,y_F)
data_pickle=pickle.load(open("../data/image_data_cross.pkl","rb"))
X_cross=data_pickle[0].astype(float)
y_cross=data_pickle[1]
print(7)
data_pickle=pickle.load(open("../data/image_data_test.pkl","rb"))
X_test=data_pickle[0].astype(float)
y_test=data_pickle[1]
print(8)
del(data_pickle)

# normalize data
Xmean=np.mean(X_train);
print(9)
Xstd=np.std(X_train);
print(10)
X_train=(X_train-Xmean)/Xstd;
print(11)
X_cross=(X_cross-Xmean)/Xstd;
print(12)
X_test=(X_test-Xmean)/Xstd;

print(13)
    
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
