import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

###################################### Define the FUNCTIONS #####################################################
#
# Create a function to fit the data to a Gaussian given some initial values
def gaussian(x,mu,sigm,amp):
    '''
    Gaussian distribution
    
    x - values for the fit
    p[0]: mu - mean of the distribution
    p[1]: sigma - stddev
    p[2]: amplitude
    '''
    return amp*np.exp(-(x-mu)**2/(2*sigm**2))

def linear(x,slope,intc):
    '''
    Linear equation
    '''
    y = slope*x + intc
    return y

# Function to create the gaussian and the linear fit
def twogaussian(x,slop,intc,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1):
    '''
    Function to fit 2 lines to a gaussian + linear.

    The parameters to introduce have to be the initial guesses. 
    x - values for the fit
    params: 1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    y = np.zeros_like(x)
    y = y + (slop*x+intc)
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1)
    return y


# Function to create the gaussian and the linear fit
def funcSII2comp(x,slop,intc,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_20,sig_20,amp_20,mu_21,sig_21,amp_21):
    '''
    Function to fit 2 lines to a gaussian + linear.
    The parameters to introduce have to be the initial guesses. 
    x - values for the fit
    params: 1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    y = np.zeros_like(x)
    y = y + (slop*x+intc)
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_20,sig_20,amp_20) + gaussian(x,mu_21,sig_21,amp_21)
    return y

# Function to create the gaussian and the linear fit
def funcgauslin(x,slop,intc,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4):
    '''
    Function to fit the spectra to a gaussian + linear.
    The parameters to introduce have to be the initial guesses. 
    The values will be the parameters for fitting the gaussians.
    x - values for the fit
    params: 1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    fy = np.zeros_like(x)
    fy = fy + (slop*x+intc)
    fy = fy + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_2,sig_2,amp_2) + gaussian(x,mu_3,sig_3,amp_3) + gaussian(x,mu_4,sig_4,amp_4)
    return fy

# Broad component of Halpha
def funcbroad(x,slop,intc,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4,mu_b,sig_b,amp_b):
    '''
    Function to fit the spectra to a broad Halpha component.
    The parameters to introduce have to be the initial guesses. 
    It is necesary to have made the linear fit first
    x - values for the fit
    params: 1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    y = np.zeros_like(x)
    y = y + (slop*x+intc)
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_2,sig_2,amp_2) + gaussian(x,mu_3,sig_3,amp_3) + gaussian(x,mu_4,sig_4,amp_4) + gaussian(x,mu_b,sig_b,amp_b)
    return y

# Second component of the lines
def func2com(x,slop,intc,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4,mu_20,sig_20,amp_20,mu_21,sig_21,amp_21,mu_22,sig_22,amp_22,mu_23,sig_23,amp_23,mu_24,sig_24,amp_24):
    '''
    Function to fit the lines to a second component.
    The parameters to introduce have to be the initial guesses. 
    It is necesary to have made the linear fit first
    x - values for the fit
    params: 1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    y = np.zeros_like(x)
    y = y + (slop*x+intc)
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_2,sig_2,amp_2) + gaussian(x,mu_3,sig_3,amp_3) + gaussian(x,mu_4,sig_4,amp_4) + gaussian(x,mu_20,sig_20,amp_20) + gaussian(x,mu_21,sig_21,amp_21) + gaussian(x,mu_22,sig_22,amp_22) + gaussian(x,mu_23,sig_23,amp_23) + gaussian(x,mu_24,sig_24,amp_24)
    return y

# Second component + broad Halpha of the lines
def func2bcom(x,slop,intc,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4,mu_20,sig_20,amp_20,mu_21,sig_21,amp_21,mu_22,sig_22,amp_22,mu_23,sig_23,amp_23,mu_24,sig_24,amp_24,mu_b,sig_b,amp_b):
    '''
    Function to fit the lines to a second component + a broad Halpha component.
    The parameters to introduce have to be the initial guesses. 
    It is necesary to have made the linear fit first
    x - values for the fit
    params: 1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    y = np.zeros_like(x)
    y = y + (slop*x+intc)
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_2,sig_2,amp_2) + gaussian(x,mu_3,sig_3,amp_3) + gaussian(x,mu_4,sig_4,amp_4) + gaussian(x,mu_20,sig_20,amp_20) + gaussian(x,mu_21,sig_21,amp_21) + gaussian(x,mu_22,sig_22,amp_22) + gaussian(x,mu_23,sig_23,amp_23) + gaussian(x,mu_24,sig_24,amp_24) + gaussian(x,mu_b,sig_b,amp_b)
    return y

