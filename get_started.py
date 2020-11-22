#%% Import some modules

import os, re, glob
os.chdir('/Users/franciscolima/Documents/Projects/metab_conv/')
import pandas as pd
import numpy as np
#import netCDF4 as nc
from matplotlib import pyplot as plt
from skimage import io
from pyms.GCMS.IO.ANDI import ANDI_reader
# TODO: Check diff between int_matrix and int_matrix_i
from pyms.IntensityMatrix import build_intensity_matrix
from pyms.Noise.SavitzkyGolay import savitzky_golay
from pyms.TopHat import tophat

#%% Download CDFs from Tohge's study

ftp_url = r'ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS528/*.cdf'
# Put wget command together, download to data/
cmd = 'wget ' + ftp_url + ' -P data/'
# Initiate download process
if not os.path.exists('data/'):
    os.system(cmd) # NOT RUN

#%% Try example w/ noise and baseline correction
# https://pymassspec.readthedocs.io/en/master/

# List all CDFs
fpaths = glob.glob('data/*.cdf')
pttn = re.compile(r'[0-9]+_[0-9]+')

# Create dir images/
if not os.path.exists('images/'):
    os.mkdir('images/')

print('Processing and saving images...')
# Iterate
for f in fpaths:
    smp_id = re.findall(pttn, string=f)[0]
    #ds = nc.Dataset(fpath)
    dat = ANDI_reader(f)
    im = build_intensity_matrix(dat)
    n_scan, n_mz = im.size
    # Noise and baseline corrections
    for i in range(n_mz):
        ic = im.get_ic_at_index(i)
        ic1 = savitzky_golay(ic)
        ic_smooth = savitzky_golay(ic1)
        ic_base = tophat(ic_smooth, struct='1.5m')
        im.set_ic_at_index(i, ic_base)

    log_im = np.log(im.intensity_array+1)
    # Avoid lossy compression!
    io.imsave('images/' + smp_id + '.png', arr=log_im)
