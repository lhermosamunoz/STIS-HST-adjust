'''
This script makes a gaussian fit to the emission lines of AGN spectra
It is needed a path, the spectrum in which the fit is going to be made & n initial estimation of the fit
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import usefunctions
from pyspeckit import mpfit


######################### Define the PATHS to the data and extract the spectra ###################################
#
#path      = input('Path to the data fits? (ex. "/mnt/data/lhermosa/HLA_data/NGC.../o.../ext_spec_crop.fits"): ')
hdulist   = fits.open('/mnt/data/lhermosa/HLA_data/NGC3245/O57205030_STISCCD_G750M/ext_spec_combin_crop.fits')	#path)	# Open the fit file to read the information
#hdulist   = fits.open('/mnt/data/lhermosa/HLA_data/NGC4736/o67110040_STISCCD_G750M/ext_spec.fits')	#path)	# Open the fit file to read the information
hdu       = hdulist[0]			# Extract the extension in which the spectra is saved
data      = hdu.data			# Save the data (i.e. the values of the flux per pixel)
data_head = hdu.header			# Save the header of the data
hdulist.close()				# Close the file as we don't need it anymore


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


#
# Start the parameters for the LINEAR fit
slope = 0.
intc  = data_cor[0]


###################################### Start the fit and the MODEL ###############################################

# Start the guesses 
p0    = [mu_0,sig_0,amp_0]
p3    = [slope,intc,mu_0,sig_0,amp_0,mu_1,sig_0,amp_1]#,mu_2, sig_0,amp_2]

fa  = {'x':newx1,'y':newy1}
fa3 = {'x':l,'y':data_cor}



# Constrains to the data using the keyword PARINFO in MPFIT
'''
 PARINFO Example:
 parinfo = [{'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]*5
 parinfo[0]['fixed'] = 1
 parinfo[4]['limited'][0] = 1
 parinfo[4]['limits'][0]  = 50.
 values = [5.7, 2.2, 500., 1.5, 2000.]
 for i in range(5): parinfo[i]['value']=values[i]

Take into account that the previous example leads to the same reference for all dictionaries
So the following trick is needed to use it
'''
nparams    = 8
parinf     = []
dictionary = {}
for i in xrange(0,nparams):		        # To initialize all the individual dictionaries separately
    dictionary['data'] = p3[i]       # We give to each dictionary the initial values for each parameter
    parinf.append(dictionary.copy())

parinf[6]['tied'] = p3[3]                   # Put constraint that the sigma should be the same for both gaussians
parinf[5]['tied'] = p3[2]*(l_SII_1/l_SII_2) # Put a constraint on the differences between wavelengths of the lines


# Make the fit using mpfit for one gaussian and for the combination of as many gaussians as wanted with
# a linear fit for the continuum
m  = mpfit.mpfit(usefunctions.gausfunct, p0,functkw=fa)
m3 = mpfit.mpfit(usefunctions.gaus3funct, p3,functkw=fa3,parinfo = parinf)

# Calculate the residuals of the data
resid = newy1 - usefunctions.gaussian(newx1,m.params)

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
plt.plot(newx1,usefunctions.gaussian(newx1,m.params),'g--')
frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
plt.plot(l,usefunctions.funcgauslin(l,m3.params),'r--')
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)')
plt.xlim(l[0],l[-1])

# Residual plot
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.plot(newx1,resid,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)')
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--')	# 3 sigma down limit

