#%% Import vars and modules

NUM_SCANS = 4200
STRUCT = '0.5m'

import os, re, glob, sys
os.chdir('/Users/franciscolima/Documents/Projects/metab_conv/')
import pandas as pd
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from matplotlib import pyplot as plt
from pyms.GCMS.IO.ANDI import ANDI_reader
from pyms.IntensityMatrix import build_intensity_matrix_i
from pyms.Noise.SavitzkyGolay import savitzky_golay_im
from pyms.TopHat import tophat_im

#%% Download CDFs from Tohge's study

ftp_url = r'ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS528/*.cdf'
# Put wget command together, download to data/
cmd = 'wget ' + ftp_url + ' -P data/'
# Initiate download process
if not os.path.exists('data/'):
    os.system(cmd) # NOT RUN

#%% Apply noise and baseline correction, save as .npy
# https://pymassspec.readthedocs.io/en/master/

# List all CDFs
fpaths = glob.glob('data/*.cdf') # N = 216

# Create dir images/
if not os.path.exists('arrays/'):
    os.mkdir('arrays/')

# http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# Define function
def cdf_processing(fpath):
    with suppress_stdout():
        # Read CDF
        dat = ANDI_reader(f)
        # Trim first NUM_SCAN timepoints
        dat.trim(1, NUM_SCANS-1)
        # Construct int matrix ~ (n_scan, n_mz)
        im = build_intensity_matrix_i(dat)
        # Basic noise and baseline corrections TODO: variance stabilization e.g. sqrt
        smooth = savitzky_golay_im(im)
        norm = tophat_im(smooth, struct=STRUCT)
    return norm

print('Processing and saving 2D arrays...')

# Define regex to extract ID and iterate
pttn = re.compile(r'[0-9]+_[0-9]+')
for f in tqdm(fpaths):
    smp_id = re.findall(pttn, string=f)[0]
    arr = cdf_processing(f)
    # Write to .npy
    np.save('arrays/' + smp_id, arr=arr.intensity_array)
