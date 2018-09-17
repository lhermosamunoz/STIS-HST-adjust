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

This is a package similar to mpfit but seems to work better in python. In the code the adjustment is made with the lmpfit. Again, it can be used either downloading the file from the github repository, or  simply with pip install lmfit, as it is a pure python package.

The useful thing here is that the parameters can be named and the code automatically differenciate between the constants and the variables. Named parameters can be held fixed or freely adjusted in the fit, or held between lower and upper bounds. In addition, parameters can be constrained as a simple mathematical expression of other Parameters.

Use pretty_print() for customizing the output of the fit and print it in an easiest way to see in the terminal.
    oneline (bool, optional) – If True prints a one-line parameters representation (default is False).
    colwidth (int, optional) – Column width for all columns specified in columns.
    precision (int, optional) – Number of digits to be printed after floating point.
    fmt ({'g', 'e', 'f'}, optional) – Single-character numeric formatter. Valid values are: ‘f’ floating point, ‘g’ floating point and  exponential, or ‘e’ exponential.
    columns (list of str, optional) – List of Parameter attribute names to print.
