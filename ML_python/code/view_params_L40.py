import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import matplotlib.pyplot as plt

clf = load('../params/fit_params_nn_image_L40_seed114_alpa1e0_m5040.joblib')

params=clf.coefs_

fig=plt.figure()
for i in range(40):
    img=params[0][:,i].reshape((480,480))
    plt.subplot(5,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,interpolation='nearest')
plt.tight_layout()
plt.subplots_adjust(wspace=0.04, hspace=0.04)
fig.set_size_inches(8,5)
plt.savefig('params_image_L40.png',bbox_inches='tight')

plt.show()
