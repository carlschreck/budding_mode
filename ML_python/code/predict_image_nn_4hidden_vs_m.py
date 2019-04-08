import time
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

# set timer
start_time = time.time()
# set up feature & label sizes 
input_layer_size=230400;  # 480x480 Input Images of Digits
hidden_layer_size=4;      # 2 hidden units
num_labels=3;             # 3 labels 

# load pickle data - training set split into 3
data_pickle=pickle.load(open("../data/image_data_train_A.pkl","rb"))
X_A=data_pickle[0].astype(float)
y_A=data_pickle[1]
data_pickle=pickle.load(open("../data/image_data_train_B.pkl","rb"))
X_B=data_pickle[0].astype(float)
y_B=data_pickle[1]
data_pickle=pickle.load(open("../data/image_data_train_C.pkl","rb"))
X_C=data_pickle[0].astype(float)
y_C=data_pickle[1]
data_pickle=pickle.load(open("../data/image_data_train_D.pkl","rb"))
X_D=data_pickle[0].astype(float)
y_D=data_pickle[1]
data_pickle=pickle.load(open("../data/image_data_train_E.pkl","rb"))
X_E=data_pickle[0].astype(float)
y_E=data_pickle[1]
data_pickle=pickle.load(open("../data/image_data_train_F.pkl","rb"))
X_F=data_pickle[0].astype(float)
y_F=data_pickle[1]
X_train=np.concatenate((X_A,X_B,X_C,X_D,X_E,X_F),axis=0)
y_train=np.concatenate((y_A,y_B,y_C,y_D,y_E,y_F),axis=0)
del(X_A,X_B,X_C,X_D,X_E,X_F,y_A,y_B,y_C,y_D,y_E,y_F)
data_pickle=pickle.load(open("../data/image_data_cross.pkl","rb"))
X_cross=data_pickle[0].astype(float)
y_cross=data_pickle[1]
data_pickle=pickle.load(open("../data/image_data_test.pkl","rb"))
X_test=data_pickle[0].astype(float)
y_test=data_pickle[1]
del(data_pickle)

# normalize data
Xmean=np.mean(X_train);
Xstd=np.std(X_train);
X_train=(X_train-Xmean)/Xstd;
X_cross=(X_cross-Xmean)/Xstd;
X_test=(X_test-Xmean)/Xstd;

# train neural network
m=5040;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m5040.joblib')
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

# train neural network
m=3600;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m3600.joblib')
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

# train neural network
m=2400;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m2400.joblib')
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

# train neural network
m=1800;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m1800.joblib')
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

# train neural network
m=1200;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m1200.joblib')
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

# train neural network
m=800;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m800.joblib')
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

# train neural network
m=600;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m600.joblib')
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

# train neural network
m=400;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m400.joblib')
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

# train neural network
m=200;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m200.joblib')
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

# train neural network
m=100;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params__nn_image_L4_seed1_alpa1e0_m100.joblib')
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

# train neural network
m=50;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params_nn_image_L4_seed1_alpa1e0_m50.joblib')
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

# train neural network
m=20;
y_train=y_train[0:m];
X_train=X_train[0:m,:];        
clf=MLPClassifier(solver='lbfgs',alpha=1.0,max_iter=200,hidden_layer_sizes=hidden_layer_size,random_state=4);
clf.fit(X_train,y_train)
p_train=clf.predict(X_train);
p_cross=clf.predict(X_cross);
p_test=clf.predict(X_test);
print(sum(p_train==y_train)/len(y_train))
print(sum(p_cross==y_cross)/len(y_cross))
print(sum(p_test==y_test)/len(y_test))
print()
dump(clf, '../params/fit_params_nn_image_L4_seed1_alpa1e0_m20.joblib')
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

# print running time
print("Running time:",int(time.time()-start_time),"seconds")
