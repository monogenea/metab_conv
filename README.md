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

# Resources

Link to the [Sci Data manuscript](https://www.nature.com/articles/sdata201851).
Info about package [here](https://pymassspec.readthedocs.io/en/master/)

# TODO

* consider transcriptomics
* Variance stabilization e.g. sqrt?
* find way to align chromatograms
* Apply ECC. Here is a nice [tutorial](https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/) and the ECC [original paper](http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf).
* Affine trasnformation (in contrast to translational ECC) takes a lot longer