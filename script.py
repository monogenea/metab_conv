
# Import some modules
import pandas as pd
#import netCDF4 as nc
from pyms.GCMS.IO. import JCAMP_reader

# Try example 
fpath = '/Users/franciscolima/Documents/metab_conv/data/9_3.cdf'
ds = nc.Dataset(fpath)
ds

#%%
for var in ds.variables.values():
    print(var)

test = ds['intensity_values'][:]
# %%
#from pyms.GCMS.IO.JCAMP import JCAMP_reader

#data = JCAMP_reader(jcamp_file)

#from pyms.IntensityMatrix import build_intensity_matrix_i
#im = build_intensity_matrix_i(data)