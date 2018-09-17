'''
This script makes a gaussian fit to the emission lines of AGN spectra
It is needed a path, the spectrum in which the fit is going to be made and the initial estimation of the fit
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lmfit


######################### Define the PATHS to the data and extract the spectra ###################################
#
#path      = input('Path to the data fits? (ex. "/mnt/data/lhermosa/HLA_data/NGC.../o.../ext_spec_crop.fits"): ')
hdulist   = fits.open('/mnt/data/lhermosa/HLA_data/NGC3245/O57205030_STISCCD_G750M/ext_spec_combin_crop.fits')	#path)	# Open the fit file to read the information
hdu       = hdulist[0]			# Extract the extension in which the spectra is saved
data      = hdu.data			# Save the data (i.e. the values of the flux per pixel)
data_head = hdu.header			# Save the header of the data
hdulist.close()				# Close the file as we don't need it anymore


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


# Function to create the gaussian and the linear fit
def funcgauslin(x,slope,intc,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4):
    '''
    Function to fit the spectra to a gaussian + linear.

    The parameters to introduce have to be the initial guesses for both components. 
    The first two values need to be the slope and the intercept, and then the rest 
    will be the parameters for fitting the gaussians.
    x - values for the fit
    params: The first two have to be the slope and the intercept of the linear fit
	    1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    fy = np.zeros_like(x)
    fy = fy + (slope*x+intc)
    fy  = fy + gaussian(x,mu_0,sig_0,amp_0)
    fy  = fy + gaussian(x,mu_1,sig_1,amp_1)
    fy  = fy + gaussian(x,mu_2,sig_2,amp_2)
    fy  = fy + gaussian(x,mu_3,sig_3,amp_3)
    fy  = fy + gaussian(x,mu_4,sig_4,amp_4)
    return fy


############################# Transform data and plot the spectra ################################################
#
# Due to the extraction of the spectra, the header is probably missing information about the axis corresponding
# to the wavelength of the spectra
# We made use of the STIS manual, where there is a formula to convert the x axis pixels in wavelength based on the 
# header information in the way as follows:
#			l(x) = CRVAL1+(x-CRPIX1)*CD1_1
# So now we transform the axis and we have the whole spectrum available for fitting

crval1 = data_head['CRVAL1']
crpix1 = data_head['CRPIX1']
cd1_1  = data_head['CD1_1']

xnew = np.arange(1,len(data)+1,1)	# We have to start in 1 for doing the transformation as no 0 pixel exist!! 
l    = crval1 + (xnew-crpix1)*cd1_1	# This is the wavelength range to use. The data vector contains the flux

# Plot the spectra and check that everything is ok
plt.ion()
plt.show()
plt.plot(l,data)
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Flux ($erg/s/cm^{2} / \AA$)')
plt.xlim(l[0],l[-1])


###################################################################################################################
################################################## MAIN ###########################################################
###################################################################################################################

###################################### Initial parameters needed #################################################
# Rest values of the line wavelengths 
l_Halpha = 6563.
l_NII_1  = 6548.
l_NII_2  = 6584.
l_SII_1  = 6716.
l_SII_2  = 6731.


#
# Now redefine the zone to fit
data_cor = data[2:-2]*10**14
l        = l[2:-2]
liminf   = np.where(l>6755.)[0][0]	#liminf   = np.where(l>6550.)[0][0]
limsup   = np.where(l<6780.)[0][-1]	#limsup   = np.where(l<6590.)[0][-1]
newx1    = l[liminf:limsup+1]
newy1    = data_cor[liminf:limsup+1]
# Initial guesses of the fitting parameters
sig_0 = 1.
mu_0  = newx1[np.argmax(newy1)]
amp_0 = max(newy1)

liminf   = np.where(l>6720.)[0][0]
limsup   = np.where(l<6755.)[0][-1]
newx2    = l[liminf:limsup+1]
newy2    = data_cor[liminf:limsup+1]
mu_1  = newx2[np.argmax(newy2)]
amp_1 = max(newy2)

liminf   = np.where(l>6610.)[0][0]
limsup   = np.where(l<6620.)[0][-1]
newx3    = l[liminf:limsup+1]
newy3    = data_cor[liminf:limsup+1]
mu_2  = newx3[np.argmax(newy3)]
amp_2 = max(newy3)

liminf   = np.where(l>6590.)[0][0]
limsup   = np.where(l<6600.)[0][-1]
newx4    = l[liminf:limsup+1]
newy4    = data_cor[liminf:limsup+1]
mu_3  = newx4[np.argmax(newy4)]
amp_3 = max(newy4)

liminf   = np.where(l>6500.)[0][0]
limsup   = np.where(l<6580.)[0][-1]
newx5    = l[liminf:limsup+1]
newy5    = data_cor[liminf:limsup+1]
mu_4  = newx5[np.argmax(newy5)]
amp_4 = max(newy5)

#
# Start the parameters for the LINEAR fit
slope = 0.
intc  = data_cor[0]


###################################### Start the fit and the MODEL ###############################################

# Put the constrains to each of the parameters in the fit, then make the fit using lmfit
# for one gaussian and for the combination of as many gaussians as wanted with a linear fit
# for the continuum


# Calculate the residuals of the data


# In order to determine if the lines need one more gaussian to be fit correctly, we apply the condition
# that the std dev of the continuum should be higher than 3 times the std dev of the residuals of the 
# fit of the line. We have to calculate the stddev of the continuum in a place where there are no 
# lines (True for all AGNs spectra in this range).
# Calculate the standard deviation of a part of the continuum without lines nor contribution of them
std0   = np.where(l>6450.)[0][0]
std1   = np.where(l<6500.)[0][-1]
stadev = np.std(data_cor[std0:std1])


#############################################################################################################
######################################## RESULTS: PLOT and PRINT ############################################
#############################################################################################################
#
#################### Plot the results ##########################
plt.close()
# MAIN plot
fig1   = plt.figure(1)
frame1 = fig1.add_axes((.1,.3,.8,.6)) #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(newx1,newy1,'k-')		# Selected data to do the fit
plt.plot(newx1,gaussian(newx1,resu.params['mu'],resu.params['sigm'],resu.params['amp']),'g--')
frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
plt.plot(l,funcgauslin(l,resu1.params['mu'],resu1.params['sigm'],resu1.params['amp']),'r--')
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)')
plt.xlim(l[0],l[-1])

# Residual plot
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.plot(newx1,resid,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)')
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') # 3 sigma down limit
