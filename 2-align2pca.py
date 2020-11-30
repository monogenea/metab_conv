#%% Import vars and modules

DIR = '/Users/franciscolima/Documents/Projects/metab_conv/'
# Pars
NITER = 2500
T_EPS = 1e-6
HEIGHT = 1024
WIDTH = 512
SEED = 999

# Imports
import os, re, glob, sys, cv2
os.chdir(DIR)
import img_utils
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#%% Load metadata

metadata = pd.read_csv(DIR + 'metadata/a_metabolomics_Fernie.txt', sep='\t')

#%% Load .npy arrays, image aligment to the smallest

# List all arrays
arrays = glob.glob(r'arrays/*.npy')

# Take avg arr as reference for alignment; for smallest arr use arrays.pop(sizes.argmin())
ref = np.mean(np.stack([img_utils.img_load_prep(arr, (WIDTH, HEIGHT)) for arr in arrays]), axis=0)

# Apply alignment over all images, add ref to list
print('ECC image alignment...')
aligned = [img_utils.img_registration(ref.astype(np.float32), arr, NITER, T_EPS) for arr in tqdm(arrays)]

# Prepare control set to compare with vs without alignment
print('Produce non-aligned counterpart for comparison...')
ctrl = [img_utils.img_load_prep(arr, (WIDTH, HEIGHT)) for arr in tqdm(arrays)]

#%% Compare global 2D variance pre- and post-alignment
var_aligned = np.var(np.stack(aligned), axis=0)
var_ctrl = np.var(np.stack(ctrl), axis=0)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
imgs = []
imgs.append(axs[0].imshow(var_ctrl))
axs[0].set_title('Non-aligned')
imgs.append(axs[1].imshow(var_aligned))
axs[1].set_title('Aligned')
cbr = fig.colorbar(imgs[0], ax=axs)
cbr.ax.set_title('Var')
fig.savefig('figures/2Dvar.png', dpi=500)
fig.clear()
#%% Principal component analysis

# Import and preprocess all samples (.npy arrays)
X_flat = np.vstack([f.flatten() for f in aligned]) # (216, WIDTH*HEIGHT)

# Mean-center pixels and perform PCA
X = StandardScaler(with_std=False).fit_transform(X_flat)
pca = PCA(n_components=2, random_state=SEED)
scores = pca.fit_transform(X)
loadings = pca.components_.T

if not os.path.exists('figures/'):
    os.mkdir('figures/')

# Extract order of arrays
pttn = re.compile(r'[0-9]+_[0-9]+')
# Regex to get sample ID
ids = [re.findall(pttn, string=name)[0] for name in arrays]
# Another regex to facilitate ranking by sample order
ids = np.array([re.sub(pattern='_', repl='.', string=i) for i in ids], dtype=np.float32)
sample_order = ids.argsort() + 1

# Plot PC 1-2
plt.scatter(scores[:,0], scores[:,1], c=sample_order)
plt.xlabel('PC1')
plt.ylabel('PC2')
cbr = plt.colorbar()
cbr.ax.set_title('Sample order')
plt.savefig('figures/pca.png', dpi=500)

# Plot loadings after reshaping
loads1 = loadings[:,0].reshape((HEIGHT, WIDTH))
loads2 = loadings[:,1].reshape((HEIGHT, WIDTH))

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
imgs = []
imgs.append(axs[0].imshow(loads1))
axs[0].set_title('Loadings 1')
imgs.append(axs[1].imshow(loads2))
axs[1].set_title('Loadings 2')
cbr = fig.colorbar(imgs[0], ax=axs)
plt.savefig('figures/loadings.png', dpi=500)
fig.clear()