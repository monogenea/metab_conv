#%% Import vars and modules

DIR = '/Users/franciscolima/Documents/Projects/metab_conv/'
# Pars
STRUCT = '0.5m'

# Imports
import os, re, glob
os.chdir(DIR)
import img_utils
import numpy as np
from tqdm import tqdm

#%% Apply noise and baseline correction, save as .npy

# List all CDFs
fpaths = glob.glob('data/*.cdf') # n=216

print('Processing and saving 2D arrays...')
# Define regex to extract ID and iterate
pttn = re.compile(r'[0-9]+_[0-9]+')
for f in tqdm(fpaths):
    smp_id = re.findall(pttn, string=f)[0]
    arr = img_utils.cdf_processing(f, structure=STRUCT) # nscans=NUM_SCANS
    # Write to .npy
    np.save('arrays/' + smp_id, arr=arr.intensity_array)
