import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

######################### Define the PATHS to the data and extract the spectra ###################################
#
hdulist   = fits.open('/mnt/data/lhermosa/HLA_data/NGC3245/O57205030_STISCCD_G750M/ext_spec_combin_crop.fits')	# Open the fit file to read the information
hdu       = hdulist[0]			# Extract the extension in which the spectra is saved
data      = hdu.data			# Save the data (i.e. the values of the flux per pixel)
data_head = hdu.header			# Save the header of the data
hdulist.close()				# Close the file as we don't need it anymore


###################################### Define the FUNCTIONS #####################################################
#
# Create a function to fit the data to a Gaussian given some initial values
def gaussian(x,p):
    '''
    Gaussian distribution
    
    x - values for the fit
    p[0]: mu - mean of the distribution
    p[1]: sigma - stddev
    p[2]: amplitude
    '''
    return p[2]*np.exp(-(x-p[0])**2/(2*p[1]**2))


# Function to create the gaussian and the linear fit
def func(x, *params):
    '''
    Function to fit the spectra to a gaussian + linear.

    The parameters to introduce have to be the initial guesses for both components. 
    The first two values need to be the slope and the intercept, and then the rest 
    will be the parameters for fitting the gaussians.
    x - values for the fit
    params: The first two have to be the slope and the intercept of the linear fit
	    1. amplitude
	    2. mu - mean of the distribution
	    3. sigma - stddev
    '''
    y = np.zeros_like(x)
    y = y + (params[0]*x+params[1])
    for i in range(2, len(params), 3):
        amp  = params[i]
        mu   = params[i+1]
        sig  = params[i+2]
        y = y + amp*np.exp(-(x-mu)**2/(2*sig**2))
    return y


# Function to calculate the chi-square of the data
def calc_reduced_chi_square(fit, x, y, yerr, N, n_free):
    '''
    fit (array) values for the fit
    x,y,yerr (arrays) data
    N total number of points
    n_free number of parameters we are fitting
    '''
    return 1.0/(N-n_free)*sum(((fit - y)/yerr)**2)


####################################################################################################################
############################# Transform data and plot the initial spectra ##########################################
####################################################################################################################
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

ocnt = input('Do you want to continue? (Y/N): ')
if ocnt == 'N': print('The initial spectra is not as expected, so dont trust in the results of the fit')


###################################################################################################################
################################################## MAIN ###########################################################
###################################################################################################################

# Rest values of the line wavelengths 
l_Halpha = 6563.
l_NII_1  = 6548.
l_NII_2  = 6584.
l_SII_1  = 6716.
l_SII_2  = 6731.

############################## Start the parameters for the fits ###############################################
# 
# Now redefine the zone to fit as the borders may contain some unwanted features due to the STIS image
data_cor = data[3:-2]*10**14
l        = l[3:-2]

#
# Start the parameters for the GAUSSIAN fit
liminf   = np.where(l>6500.)[0][0]
limsup   = np.where(l<6580.)[0][-1]
newx1    = l[liminf:limsup+1]
newy1    = data_cor[liminf:limsup+1]
# Initial guesses of the fitting parameters
sig_0 = 0.6
mu_0  = newx1[np.argmax(newy1)]
amp_0 = max(newy1)

liminf   = np.where(l>6590.)[0][0]
limsup   = np.where(l<6600.)[0][-1]
newx2    = l[liminf:limsup+1]
newy2    = data_cor[liminf:limsup+1]
mu_1  = newx2[np.argmax(newy2)]
amp_1 = max(newy2)

liminf   = np.where(l>6610.)[0][0]
limsup   = np.where(l<6660.)[0][-1]
newx3    = l[liminf:limsup+1]
newy3    = data_cor[liminf:limsup+1]
mu_2  = newx3[np.argmax(newy3)]
amp_2 = max(newy3)

liminf   = np.where(l>6720.)[0][0]
limsup   = np.where(l<6755.)[0][-1]
newx4    = l[liminf:limsup+1]
newy4    = data_cor[liminf:limsup+1]
mu_3  = newx4[np.argmax(newy4)]
amp_3 = max(newy4)

liminf   = np.where(l>6755.)[0][0]
limsup   = np.where(l<6780.)[0][-1]
newx5    = l[liminf:limsup+1]
newy5    = data_cor[liminf:limsup+1]
mu_4  = newx5[np.argmax(newy5)]
amp_4 = max(newy5)

#
# Start the parameters for the LINEAR fit
slope = 0.
intc  = data_cor[0]

############################################### FIT #########################################################

#
# We implement the least-square fitting program below the Levenberg-Marquardt method

# Initial guesses of the fitting parameters
guess    = [slope,intc,amp_0,mu_0,sig_0,amp_1,mu_1,sig_0,amp_2,mu_2,sig_0,amp_3,mu_3,sig_0,amp_4,mu_4,sig_0]  

# Calculate the best parameter values for the fit and then apply them to the function
popt, pcov = curve_fit(func, l, data_cor, p0=guess)
fit        = func(l, *popt)

# Calculate the errors associated to the fit
perr = np.sqrt(np.diag(pcov))

# Calculate the residuals of the fit
resid = data_cor - fit

# Calculate the standard deviation of a part of the continuum without lines nor contribution of them
std0   = np.where(l>6450.)[0][0]
std1   = np.where(l<6500.)[0][-1]
stadev = np.std(data_cor[std0:std1])

#############################################################################################################
######################################## RESULTS: PLOT and PRINT ############################################
#############################################################################################################
#
# Now we create the individual gaussians in order to plot and print the results
n = 0
print('				RESULTS OF THE FIT: ')
print('Linear fit equation: {:.5f}*x + {:.5f} with errors +/- {:.5f} and +/- {:.5f}'.format(popt[0], popt[1], perr[0], perr[1]))
for i in range(2, len(popt), 3):
    n += 1
    amp  = popt[i]
    mu   = popt[i+1]
    sig  = popt[i+2]
    erramp = perr[i]
    errmu  = perr[i+1]
    errsig = perr[i+2]
    print('Gaussian n'+str(n)+ ' parameters: Amplitude: {:.5f} +/- {:.5f} /// MEAN: {:.3f} +/- {:.3f} /// STANDARD DEVIATION: {:.5f} +/- {:.5f}'.format(amp, erramp, mu, errmu, sig, errsig))

# Now we create and plot the individual gaussians of the fit
gaus1 = gaussian(l,[popt[3],popt[4],popt[2]]) 
gaus2 = gaussian(l,[popt[6],popt[7],popt[5]])
gaus3 = gaussian(l,[popt[9],popt[10],popt[8]])
gaus4 = gaussian(l,[popt[12],popt[13],popt[11]])
gaus5 = gaussian(l,[popt[15],popt[16],popt[14]])

################################################ PLOT ######################################################
plt.close()

# MAIN plot
fig1   = plt.figure(1)
frame1 = fig1.add_axes((.1,.3,.8,.6)) #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)
plt.plot(l, fit , 'r-')		# MAIN fit
plt.plot(l,gaus1,'k--',label='Gauss 1')
plt.plot(l,gaus2,'g--',label='Gauss 2')
plt.plot(l,gaus3,'c--',label='Gauss 3')
plt.plot(l,gaus4,'y--',label='Gauss 4')
plt.plot(l,gaus5,'m--',label='Gauss 5')
plt.plot(l,(popt[0]*l+popt[1]),'k-.',label='Linear fit')
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
plt.legend()
frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)')
plt.xlim(l[0],l[-1])

# Residual plot
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.plot(l,np.zeros(len(l)),'k--')	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--')	# 3 sigma down limit
for i in range(2, len(popt), 3):	# This is to mark the position of the center of gaussians in the residual fit
    mu   = popt[i+1]
    plt.plot(mu+np.zeros(len(resid)),resid,color='pink')

plt.plot(l,resid,color='grey')		# Main
plt.xlim(l[0],l[-1])
plt.xlabel('Wavelength ($\AA$)')

