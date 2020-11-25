#%% Import vars and modules
# TODO: consider transcriptomics

DIR = '/Users/franciscolima/Documents/Projects/metab_conv/'
TARGET_SIZE = (512, 256)

import os, re, glob, sys, cv2
os.chdir(DIR)
import pandas as pd
import numpy as np
#from tqdm import tqdm
#from contextlib import contextmanager
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#%% PCA of preprocess images

def img_prep(fpath):
    arr = np.load(fpath)
    arr = cv2.blur(arr, ksize=(3,3))
    arr = cv2.resize(arr, dsize=TARGET_SIZE)
    return arr

list_arr = glob.glob(r'arrays/*.npy')

# Import and preprocess all samples (.npy arrays)
all_dat = np.vstack([img_prep(f).flatten() for f in list_arr]) # (216, 131072)

# Mean-center pixels and perform PCA
X = StandardScaler(with_std=False).fit_transform(all_dat)
mod = PCA().fit_transform(X)

if not os.path.exists('figures/'):
    os.mkdir('figures/')

# Plot PC 1-2
plt.scatter(mod[:,0], mod[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('figures/pca.png', dpi=500)