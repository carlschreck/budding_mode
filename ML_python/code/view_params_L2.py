import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import matplotlib.pyplot as plt

clf = load('../params/fit_params_nn_image_L2_seed1_alpa1e0.joblib')

params=clf.coefs_
params0=params[0][:,0]
params1=params[0][:,1]

img0=params0.reshape((480,480))
img1=params1.reshape((480,480))


fig = plt.figure()

plt.subplot(1, 2, 1)
plt.xticks([])
plt.yticks([])
imgplot=plt.imshow(img0,interpolation='nearest')

plt.subplot(1, 2, 2)
plt.xticks([])
plt.yticks([])
imgplot=plt.imshow(img1,interpolation='nearest')
plt.tight_layout()
plt.savefig('../figures/params_image_L2.png')

plt.show()
