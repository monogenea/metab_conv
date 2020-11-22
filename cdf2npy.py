#%% Import some modules

# TODO: Check diff between int_matrix and int_matrix_i
# TODO: Variance stabilization e.g. sqrt

#import netCDF4 as nc
import os, re, glob
os.chdir('/Users/franciscolima/Documents/Projects/metab_conv/')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pyms.GCMS.IO.ANDI import ANDI_reader
from pyms.IntensityMatrix import build_intensity_matrix
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
pttn = re.compile(r'[0-9]+_[0-9]+')

# Create dir images/
if not os.path.exists('arrays/'):
    os.mkdir('arrays/')

print('Processing and saving 2D arrays...')

# Iterate
for f in fpaths:
    smp_id = re.findall(pttn, string=f)[0]
    dat = ANDI_reader(f)
    im = build_intensity_matrix(dat) # (n_scan, n_mz)

    # Basic noise and baseline corrections
    smooth = savitzky_golay_im(im)
    norm = tophat_im(smooth, struct='1m')
    
    # Write to .npy
    np.save('arrays/' + smp_id, arr=norm.intensity_array)
