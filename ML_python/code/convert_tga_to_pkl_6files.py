import sys
import time
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# plot 1 example
img=Image.open('../production/imaging/axial/frames/frame1.tga')
imgplot=plt.imshow(img)
plt.show()

# set timer
start_time = time.time()

# fraction of data in training set
frac_train=0.7;
frac_cross=0.15;

# set up feature & label sizes 
seedmax=2400;       # number of sets per label
num_pixels=230400;  # 480x480 input image
num_labels=3;       # 3 labels 

# set timer
prev_time=start_time

# read in images from 3 directories
count=0
X=np.zeros((num_labels*seedmax,num_pixels));
y=np.zeros(num_labels*seedmax);
dir='../production/imaging/axial/frames/';
for seed in range(1,seedmax+1):
    I=Image.open(dir+'frame'+str(seed)+'.tga');
    X[count,:]=[line[0] for line in list(I.getdata())]
    y[count]=1
    count+=1         
dir='../production/imaging/polar1/frames/';
for seed in range(1,seedmax+1):
    I=Image.open(dir+'frame'+str(seed)+'.tga');
    X[count,:]=[line[0] for line in list(I.getdata())]
    y[count]=2    
    count+=1   
dir='../production/imaging/random/frames/';
for seed in range(1,seedmax+1):
    I=Image.open(dir+'frame'+str(seed)+'.tga');
    X[count,:]=[line[0] for line in list(I.getdata())]
    y[count]=3    
    count+=1    
              
# size of training & cross validation sets
tot_sets=seedmax*num_labels
len_train=round(frac_train*tot_sets)
len_cross=round(frac_cross*tot_sets)
len_test=tot_sets-len_train-len_cross

# break into training, cross validation, & test sets
randnum=np.random.rand(len(y))
randind=np.argsort(randnum)
X_train=X[randind[0:len_train],:].astype(int)
X_cross=X[randind[len_train:len_train+len_cross],:].astype(int)
X_test=X[randind[len_train+len_cross:],:].astype(int)
y_train=y[randind[0:len_train]].astype(int)
y_cross=y[randind[len_train:len_train+len_cross]].astype(int)
y_test=y[randind[len_train+len_cross:]].astype(int)

# pickle (save) data
len_save=int(len_train/6) # break training set into six
data_pickle=[X_train[0:len_save,:],y_train[0:len_save]]
pickle.dump(data_pickle,open("image_data_train_A.pkl","wb"))
data_pickle=[X_train[len_save:2*len_save,:],y_train[len_save:2*len_save]]
pickle.dump(data_pickle,open("image_data_train_B.pkl","wb"))
data_pickle=[X_train[2*len_save:3*len_save,:],y_train[2*len_save:3*len_save]]
pickle.dump(data_pickle,open("image_data_train_C.pkl","wb"))
data_pickle=[X_train[3*len_save:4*len_save,:],y_train[3*len_save:4*len_save]]
pickle.dump(data_pickle,open("image_data_train_D.pkl","wb"))
data_pickle=[X_train[4*len_save:5*len_save,:],y_train[4*len_save:5*len_save]]
pickle.dump(data_pickle,open("image_data_train_E.pkl","wb"))
data_pickle=[X_train[5*len_save:,:],y_train[5*len_save:]]
pickle.dump(data_pickle,open("image_data_train_F.pkl","wb"))
data_pickle=[X_cross,y_cross]
pickle.dump(data_pickle,open("image_data_cross.pkl","wb"))
data_pickle=[X_test,y_test]
pickle.dump(data_pickle,open("image_data_test.pkl","wb"))

# print running time
print("Running time:",int(time.time()-start_time),"seconds")

