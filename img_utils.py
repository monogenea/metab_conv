#%% Utils to process images

# Imports
import os, sys, cv2
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from matplotlib import pyplot as plt
from pyms.GCMS.IO.ANDI import ANDI_reader
from pyms.IntensityMatrix import build_intensity_matrix_i
from pyms.Noise.SavitzkyGolay import savitzky_golay_im
from pyms.TopHat import tophat_im

#%% CDF processing w/ baseline correction and noise smoothing
# http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
@contextmanager
def suppress_stdout(): # suppress verbose
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# Define function
def cdf_processing(fpath, structure): # nscans
    with suppress_stdout():
        # Read CDF
        dat = ANDI_reader(fpath)
        # Trim first NUM_SCAN timepoints
        #dat.trim(1, nscans-1)
        # Construct int matrix ~ (n_scan, n_mz)
        im = build_intensity_matrix_i(dat)
        # Basic noise and baseline corrections
        smooth = savitzky_golay_im(im)
        norm = tophat_im(smooth, struct=structure)
    return norm

#%% Image registation using Enhanced Correlation Coefficient Maximization (ECC)
def img_registration(pref, parr, niter=5000, termination_eps=1e-10):
    # Import the reference and target arrays
    ref = np.load(pref).astype(np.float32)
    arr = np.load(parr).astype(np.float32)
    sz = ref.shape
    # Define model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Set criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, niter, termination_eps)
    # Run method, return aligned array w/ shape of ref
    _, warp_matrix = cv2.findTransformECC(ref, arr, warp_matrix, warp_mode, criteria)
    aligned = cv2.warpAffine(arr, warp_matrix, (sz[1], sz[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned

#%% Image processing fun
def img_prep(fpath, target_size):
    arr = np.load(fpath)
    arr = cv2.blur(arr, ksize=(3, 3))
    arr = cv2.resize(arr, dsize=target_size)
    return arr
