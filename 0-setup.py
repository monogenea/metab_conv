#%% Setup subdirs and download data

DIR = '/Users/franciscolima/Documents/Projects/metab_conv/'

# Imports
import os
os.chdir(DIR)

# Create project subdirs
os.makedirs('data/')
os.makedirs('metadata/')
os.makedirs('arrays/')
os.makedirs('figures/')

#%% Download CDFs from Tohge's study

ftp_url = r'ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS528/*.cdf'
# Put wget command together, download to data/
cmd_cdf = 'wget ' + ftp_url + ' -P data/'
# Initiate download process
os.system(cmd_cdf)

#%% Download ISA-Tab metadata

md_url = 'https://static-content.springer.com/esm/art%3A10.1038%2Fsdata.2018.51/MediaObjects/41597_2018_BFsdata201851_MOESM8_ESM.zip'
# Put wget command together, download to metadata/, clean up
cmd_md = 'wget ' + md_url + ' -P metadata/ ; unzip metadata/*.zip -d metadata/ ; rm metadata/*.zip'
os.system(cmd_md)