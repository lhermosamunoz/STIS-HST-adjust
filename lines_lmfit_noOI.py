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

# First we have to initialise the model by doing
simp_mod   = lmfit.Model(gaussian)
comp_mod   = lmfit.Model(funcgauslin)

# Now we define the initial guesses and the constraints
params = lmfit.Parameters()
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
ab = lmfit.Parameter('slope', value=slope)
bc = lmfit.Parameter('intc', value=intc)
meth = input('Which method do you want to use? (options are S, M1 or M2): ')	# Method to fit: S/O/M1/M2
if meth == 'S':
    cd = lmfit.Parameter('mu_0', value=mu_0)
    de = lmfit.Parameter('sig_0', value=sig_0)
    ef = lmfit.Parameter('amp_0', value=amp_0)
    fg = lmfit.Parameter('mu_1', value=mu_1,expr='mu_0*(6716./6731.)')
    gh = lmfit.Parameter('sig_1', value=sig_1,expr='sig_0')
    hi = lmfit.Parameter('amp_1', value=amp_1)
    ij = lmfit.Parameter('mu_2', value=mu_2,expr='mu_0*(6584./6731.)')
    jk = lmfit.Parameter('sig_2', value=sig_2,expr='sig_0')
    kl = lmfit.Parameter('amp_2', value=amp_2)
    lm = lmfit.Parameter('mu_3', value=mu_3,expr='mu_0*(6563./6731.)')
    mn = lmfit.Parameter('sig_3', value=sig_3,expr='sig_0')
    no = lmfit.Parameter('amp_3', value=amp_3)
    op = lmfit.Parameter('mu_4', value=mu_4,expr='mu_0*(6548./6731.)')
    pq = lmfit.Parameter('sig_4', value=sig_4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value=amp_4,expr='amp_2*(1./3.)')
elif meth == 'M1':
    cd = lmfit.Parameter('mu_0', value=mu_0)
    de = lmfit.Parameter('sig_0', value=sig_0)
    ef = lmfit.Parameter('amp_0', value=amp_0)
    fg = lmfit.Parameter('mu_1', value=mu_1,expr='mu_0*(6716./6731.)')
    gh = lmfit.Parameter('sig_1', value=sig_1,expr='sig_0')
    hi = lmfit.Parameter('amp_1', value=amp_1)
    ij = lmfit.Parameter('mu_2', value=mu_2,expr='mu_0*(6584./6731.)')
    jk = lmfit.Parameter('sig_2', value=sig_2,expr='sig_0')
    kl = lmfit.Parameter('amp_2', value=amp_2)
    lm = lmfit.Parameter('mu_3', value=mu_3)		# It should be attached to the [OI] lines when available!!
    mn = lmfit.Parameter('sig_3', value=sig_3)		# It should be attached to the [OI] lines when available!!
    no = lmfit.Parameter('amp_3', value=amp_3)
    op = lmfit.Parameter('mu_4', value=mu_4,expr='mu_0*(6548./6731.)')
    pq = lmfit.Parameter('sig_4', value=sig_4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value=amp_4,expr='amp_2*(1./3.)')
elif meth == 'M2':
    print('No OI lines in this spectra and this method does not work. Please select another one')
    cd = lmfit.Parameter('mu_0', value=mu_0)
    de = lmfit.Parameter('sig_0', value=sig_0)
    ef = lmfit.Parameter('amp_0', value=amp_0)
    fg = lmfit.Parameter('mu_1', value=mu_1,expr='mu_0*(6716./6731.)')
    gh = lmfit.Parameter('sig_1', value=sig_1,expr='sig_0')
    hi = lmfit.Parameter('amp_1', value=amp_1)
    ij = lmfit.Parameter('mu_2', value=mu_2)				# It should be attached to the [OI] lines when available!!
    jk = lmfit.Parameter('sig_2', value=sig_2)				# It should be attached to the [OI] lines when available!!
    kl = lmfit.Parameter('amp_2', value=amp_2)				# It should be attached to the [OI] lines when available!!
    lm = lmfit.Parameter('mu_3', value=mu_3,expr='mu_2*(6563./6584.)')		# It should be attached to the [OI] lines when available!!
    mn = lmfit.Parameter('sig_3', value=sig_3,expr='sig_2')			# It should be attached to the [OI] lines when available!!
    no = lmfit.Parameter('amp_3', value=amp_3)					# It should be attached to the [OI] lines when available!!
    op = lmfit.Parameter('mu_4', value=mu_4,expr='mu_2*(6548./6584.)')		# It should be attached to the [OI] lines when available!!
    pq = lmfit.Parameter('sig_4', value=sig_4,expr='sig_2')			# It should be attached to the [OI] lines when available!!
    qr = lmfit.Parameter('amp_4', value=amp_4,expr='amp_2*(1./3.)')		# It should be attached to the [OI] lines when available!!

# add a sequence of Parameters
params.add_many(ab,bc,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr)

# and make the fit using lmfit
resu1 = comp_mod.fit(data_cor,params,x=l)

# In order to determine if the lines need one more gaussian to be fit correctly, we apply the condition
# that the std dev of the continuum should be higher than 3 times the std dev of the residuals of the 
# fit of the line. We have to calculate the stddev of the continuum in a place where there are no 
# lines (True for all AGNs spectra in this range).
# Calculate the standard deviation of a part of the continuum without lines nor contribution of them
std0   = np.where(l>6450.)[0][0]
std1   = np.where(l<6500.)[0][-1]
stadev = np.std(data_cor[std0:std1])
liminf   = np.where(l>6755.)[0][0]	# Strongest SII line
limsup   = np.where(l<6780.)[0][-1]
std_line = np.std(resu1.residual[liminf:limsup])


#############################################################################################################
######################################## RESULTS: PLOT and PRINT ############################################
#############################################################################################################
#
# Now we create the individual gaussians in order to plot and print the results
print('				RESULTS OF THE FIT: ')
print('Linear fit equation: {:.5f}*x + {:.5f}'.format(resu1.values['slope'], resu1.values['intc']))
print('')
print('The rest of the results can be displayed all together with resu1.params; the data can be accesed with resu1.values['']')
print(resu1.params['mu_0'])
print(resu1.params['sig_0'])
print(resu1.params['amp_0'])
print(resu1.params['mu_1'])
print(resu1.params['sig_1'])
print(resu1.params['amp_1'])
print(resu1.params['mu_2'])
print(resu1.params['sig_2'])
print(resu1.params['amp_2'])
print(resu1.params['mu_3'])
print(resu1.params['sig_3'])
print(resu1.params['amp_3'])
print(resu1.params['mu_4'])
print(resu1.params['sig_4'])
print(resu1.params['amp_4'])
print('')
print('The chi-square of the fit is: {:.5f}'.format(resu1.chisqr))
print('')
print('The standard deviation of the continuum is: {:.5f}  and the one of the SII line is: {:.5f}'.format(stadev, std_line))
print('The condition needs to be std_line < 3*std_cont --> '+str(std_line)+'< 3*'+str(stadev))

# Now we create and plot the individual gaussians of the fit
gaus1 = gaussian(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0']) 
gaus2 = gaussian(l,resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'])
gaus3 = gaussian(l,resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2'])
gaus4 = gaussian(l,resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'])
gaus5 = gaussian(l,resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'])

################################################ PLOT ######################################################
plt.close()
# MAIN plot
fig1   = plt.figure(1)
frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,funcgauslin(l,resu1.values['slope'],resu1.values['intc'],
		       resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0'],
		       resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'],
		       resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2'],
		       resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'],
		       resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4']),'r--')
plt.plot(l,gaus1,'k--',label='Gauss 1')
plt.plot(l,gaus2,'g--',label='Gauss 2')
plt.plot(l,gaus3,'c--',label='Gauss 3')
plt.plot(l,gaus4,'y--',label='Gauss 4')
plt.plot(l,gaus5,'m--',label='Gauss 5')
plt.plot(l,(resu1.values['slope']*l+resu1.values['intc']),'k-.',label='Linear fit')
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
plt.plot(l[liminf:limsup],data_cor[liminf:limsup],'y')	# Zone where the line stddev is calculated

frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)')
plt.xlim(l[0],l[-1])

# RESIDUAL plot
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.plot(l,resu1.residual,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Residuals')
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') # 3 sigma down limit
