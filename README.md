# STIS-HST-adjust
This is for making and adjustment to the emission lines seen is LINER-II galaxies using spectra coming from HST/STIS instrument.
This data can be downloaded from the HLA web, and the spectra has to be properly extracted and corrected for the aims of this work.
We can try to fit the data with different methods implemented in Python:
# CURVE_FIT (scipy)
This package is really useful for fitting the whole spectra but it doesn't allow the user to put constraints to the data (at least in an easy way). So at the end it is not very useful.
# MPFIT
This is a package originally implemented in IDL and translated to python. It als implements a fit to the functions that you want, but there is no clear help on how to use it with python. The parameter information argument is not well explained at least in the webpages I have searched for. It could be useful to try harder and find a solution for this. An example of one gaussian works well. It can be downloaded in a github repository and use it as mpfit.py, or it is also included in the python package "pyspeckit" (from pyspeckit import mpfit).
# LMFIT
This is a package similar to mpfit but seems to work better in python. Now I'm implementing a code in which the adjustment is made with the lmpfit to see how it works. Again, it can be used either downloading the file from the github repository, or it is included in 
