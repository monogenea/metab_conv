#%% Import vars and modules

DIR = '/Users/franciscolima/Documents/Projects/metab_conv/'
HEIGHT = 1050
WIDTH = 1901
SEED = 999

import os, re, glob, sys, cv2
os.chdir(DIR)
import img_utils
from tqdm import tqdm
import pandas as pd
import numpy as np
#from tqdm import tqdm
#from contextlib import contextmanager
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#%% Load metadata

metadata = pd.read_csv(DIR + 'metadata/a_metabolomics_Fernie.txt', sep='\t')

#%% PCA of preprocess images

arrays = glob.glob(r'arrays/*.npy')
# Extract no. of scans per arr
#sizes = [np.load(f).shape[0] for f in list_arr]

# Choose a random array
ref = arrays.pop()
# Apply translation-based alignment over all images
print('Running translation-based ECC image alignment...')
aligned_arrays = [img_utils.img_registration(ref, arr) for arr in tqdm(arrays)]

#%% PRINCIPAL COMPONENT ANALYSIS

# Import and preprocess all samples (.npy arrays)
all_dat = np.vstack([img_prep(f, (WIDTH, HEIGHT)).flatten() for f in tqdm(list_arr])) # (216, 131072)

# Mean-center pixels and perform PCA
X = StandardScaler(with_std=False).fit_transform(all_dat)
pca = PCA(n_components=2, random_state=SEED)
scores = pca.fit_transform(X)
loadings = pca.components_.T

if not os.path.exists('figures/'):
    os.mkdir('figures/')

# Plot PC 1-2
plt.scatter(scores[:,0], scores[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('figures/pca.png', dpi=500)

# Plot loadings after reshaping
loads1 = loadings[:,0].reshape((HEIGHT, WIDTH))
loads2 = loadings[:,1].reshape((HEIGHT, WIDTH))

plt.imshow(loads1)
plt.savefig('figures/loadings_1.png', dpi=500)
plt.imshow(loads2)
plt.savefig('figures/loadings_2.png', dpi=500)