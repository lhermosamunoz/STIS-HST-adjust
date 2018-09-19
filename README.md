# STIS-HST-adjust
This is for making and adjustment to the emission lines seen is LINER-II galaxies using spectra coming from HST/STIS instrument.
This data can be downloaded from the HLA web, and the spectra has to be properly extracted and corrected for the aims of this work.
We can try to fit the data with different methods implemented in Python:

  # CURVE_FIT (scipy)
This package is really useful for fitting the whole spectra but it doesn't allow the user to put constraints to the data (at least in an easy way). So at the end it is not very useful.

  # MPFIT
This is a package originally implemented in IDL and translated to python. It als implements a fit to the functions that you want, but there is no clear help on how to use it with python. The parameter information argument is not well explained at least in the webpages I have searched for. It could be useful to try harder and find a solution for this. An example of one gaussian works well.

It can be downloaded in a github repository and use it as mpfit.py, or it is also included in the python package "pyspeckit" (from pyspeckit import mpfit).

  # LMFIT
LMfit-py provides a Least-Squares Minimization routine and class with a simple, flexible approach to parameterizing a model for fitting to data.

This is a package similar to curve_fit but implemented to work also with limits and constraints in the parameters. Again, it can be used either downloading the file from the github repository (with git), or simply with pip install lmfit, as it is a pure python package.

The useful thing here is that the parameters can be named and the code automatically differenciate between the constants and the variables. Named parameters can be held fixed or freely adjusted in the fit, or held between lower and upper bounds. In addition, parameters can be constrained as a simple mathematical expression of other Parameters.

Also one can use print(lmfit.model.fit_report(result)) for seeing all the results for the parameters, the correlations and the statistics.

Now a first version works perfectly for one spectrum, that is the file not containing the [OI] lines. For the rest of the spectra, the code is under construction yet.
