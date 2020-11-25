# Explore 2D content of MS data

## CDF download and preprocessing

The raw data download is accomplished from `wget` calls to the FTP associated with the MetaboLights study *MTBLS528: The natural variance of the Arabidopsis floral secondary metabolites*. The CDFs are subsequently processed using utilities from the Python `PyMassSpec` package:

1. Selection of the first *N* scans (seemingly endpoints are not consistent)
2. Savitsky-Golay filter (profile smoothing)
3. Tophat noise filtering with a structure of length *t* minutes (baseline correction)
4. Write processed data as NumPy arrays

## Principal Component Analysis of stacked images

TODO

## Convolutional approaches

TODO