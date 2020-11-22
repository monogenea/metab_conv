#%% Import some modules

import os, glob
os.chdir('/Users/franciscolima/Documents/Projects/metab_conv/')
import pandas as pd
import numpy as np
#import netCDF4 as nc
from matplotlib import pyplot as plt
from skimage import io
from pyms.GCMS.IO.ANDI import ANDI_reader
# TODO: Check diff between int_matrix and int_matrix_i
from pyms.IntensityMatrix import build_intensity_matrix

#%% Download CDFs from Tohge's study

ftp_url = 'ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS528/\*.cdf'
# Put wget command together, download to data/
cmd = 'wget ' + ftp_url + ' -P data/'
# Initiate download process
if not os.path.exists('data/'):
    os.system(cmd) # NOT RUN

#%% Try example 

# TODO: Work on baseline correction
fpaths = glob.glob('data/*.cdf')

#ds = nc.Dataset(fpath)
dat = ANDI_reader(fpaths[10])
im = build_intensity_matrix(dat) # (nscan, nm/z) ~ (4401, 1901)
log_im = np.log(im.intensity_array+1)
# Visualize
plt.imshow(log_im)
 
#%% use skimage.io to export im

os.mkdir('images/')
# Avoid lossy compression!
io.imsave('images/' + f + '.png', arr=im)
