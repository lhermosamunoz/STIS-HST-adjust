'''
This script makes a gaussian fit to the emission lines of AGN spectra
It is needed a path, the spectrum in which the fit is going to be made and the initial estimation of the fit
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lmfit
import scipy.stats as stats
import os


######################### Define the PATHS to the data and extract the spectra ###################################
#
path  = input('Path to the data fits? (ex. "/mnt/data/lhermosa/HLA_data/NGC.../o.../"): ')
FILE  = input('Name of the spectra? (ex. "ext_spec_crop.fits"): ')
hdulist   = fits.open(path+FILE)	#'/mnt/data/lhermosa/HLA_data/NGC3245/O57205030_STISCCD_G750M/ext_spec_combin_crop.fits')	# Open the fit file to read the information
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

def linear(x,slope,intc):
    '''
    Linear equation
    '''
    y = slope*x + intc
    return y

# Function to create the gaussian and the linear fit
def twogaussian(x,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1):
    '''
    Function to fit 2 lines to a gaussian + linear.
    The parameters to introduce have to be the initial guesses. 
    x - values for the fit
    params: 1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    y = np.zeros_like(x)
    y = y + (new_slop*x+new_intc)
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1)
    return y


# Function to create the gaussian and the linear fit
def funcSII2comp(x,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_20,sig_20,amp_20,mu_21,sig_21,amp_21):
    '''
    Function to fit 2 lines to a gaussian + linear.
    The parameters to introduce have to be the initial guesses. 
    x - values for the fit
    params: 1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    y = np.zeros_like(x)
    y = y + (new_slop*x+new_intc)
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_20,sig_20,amp_20) + gaussian(x,mu_21,sig_21,amp_21)
    return y

# Function to create the gaussian and the linear fit
def funcgauslin(x,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4):
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
    fy = fy + (new_slop*x+new_intc)
    fy = fy + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_2,sig_2,amp_2) + gaussian(x,mu_3,sig_3,amp_3) + gaussian(x,mu_4,sig_4,amp_4)
    return fy

# Broad component of Halpha
def funcbroad(x,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4,mu_b,sig_b,amp_b):
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
    y = y + (new_slop*x+new_intc)
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_2,sig_2,amp_2) + gaussian(x,mu_3,sig_3,amp_3) + gaussian(x,mu_4,sig_4,amp_4) + gaussian(x,mu_b,sig_b,amp_b)
    return y

# Second component of the lines
def func2com(x,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4,mu_20,sig_20,amp_20,mu_21,sig_21,amp_21,mu_22,sig_22,amp_22,mu_23,sig_23,amp_23,mu_24,sig_24,amp_24):
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
    y = y + (new_slop*x+new_intc)
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_2,sig_2,amp_2) + gaussian(x,mu_3,sig_3,amp_3) + gaussian(x,mu_4,sig_4,amp_4) + gaussian(x,mu_20,sig_20,amp_20) + gaussian(x,mu_21,sig_21,amp_21) + gaussian(x,mu_22,sig_22,amp_22) + gaussian(x,mu_23,sig_23,amp_23) + gaussian(x,mu_24,sig_24,amp_24)
    return y

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

# Constants and STIS parameters
v_luz = 299792.458 # km/s
pix_to_v = 50	# km/s
sig_inst = 1.32	# pix
ang_to_pix = 0.56

# Now redefine the zone to fit
data_cor = data[2:-2]*10**14
l = l[2:-2]

if not os.path.exists(path+'ranges.txt'):
    l1 = input('lambda inf for SII 2 (angs)?: ')
    l2 = input('lambda sup for SII 2 (angs)?: ')
    l3 = input('lambda inf for SII 1 (angs)?: ')
    l4 = input('lambda sup for SII 1 (angs)?: ')
    l5 = input('lambda inf for NII 2 (angs)?: ')
    l6 = input('lambda sup for NII 2 (angs)?: ')
    l7 = input('lambda inf for Halpha (angs)?: ')
    l8 = input('lambda sup for Halpha (angs)?: ')
    l9 = input('lambda inf for NII 1 (angs)?: ')
    l10 = input('lambda sup for NII 1 (angs)?: ')
else:
    t = np.genfromtxt(path+'ranges.txt')
    l1 = t[0,]
    l2 = t[1,]
    l3 = t[2,]
    l4 = t[3,]
    l5 = t[4,]
    l6 = t[5,]
    l7 = t[6,]
    l8 = t[7,]
    l9 = t[8,]
    l10 = t[9,]

newx1 = l[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]+1]		# SII2
newy1 = data_cor[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]+1]
newx2 = l[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]+1]		# SII1
newy2 = data_cor[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]+1]
newx3 = l[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]+1]		# NII2
newy3 = data_cor[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]+1]
newx4 = l[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]+1]		# Halpha
newy4 = data_cor[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]+1]
newx5 = l[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]+1]	# NII1
newy5 = data_cor[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]+1]

# Initial guesses of the fitting parameters
sig0 = 1.			# SII2
sig20 = 1.5
mu0  = newx1[np.argmax(newy1)]
amp0 = max(newy1)
amp20 = max(newy1)/2.
sig1 = 1.			# SII1
sig21 = 1.5
mu1 = newx2[np.argmax(newy2)]
amp1 = max(newy2)
amp21 = max(newy2)/2.

#
# Start the parameters for the LINEAR fit
in_slope = 0.
in_intc  = data_cor[0]

# Redefine the lambda zone with the first and last point and the zones in between NII2-SII1 and SII2-final
newl = l[1]
l11 = input('Aprox max wavelength of the spectra?: ')
zone_S_fin = l[np.where(l<l2)[0][-1]+10:np.where(l>l11)[0][0]]
zone_N_S = l[np.where(l<l6)[0][-1]+10:np.where(l>l3)[0][0]-10]
newl = np.append(newl,zone_N_S)
newl = np.append(newl,zone_S_fin)
newl = np.append(newl,l[-1])
# now we do the same but with the flux data (y vector)
newflux = data_cor[1]
zon_S_fin = data_cor[np.where(l<l2)[0][-1]+10:np.where(l>l11)[0][0]]
zon_N_S = data_cor[np.where(l<l6)[0][-1]+10:np.where(l>l3)[0][0]-10]
newflux = np.append(newflux,zon_N_S)
newflux = np.append(newflux,zon_S_fin)
newflux = np.append(newflux,data_cor[-1])


####################################### Standard deviation of the continuum #############################################
#
# In order to determine if the lines need one more gaussian to be fit correctly, we apply the condition
# that the std dev of the continuum should be higher than 3 times the std dev of the residuals of the 
# fit of the line. We have to calculate the stddev of the continuum in a place where there are no 
# lines (True for all AGNs spectra in this range).
# Calculate the standard deviation of a part of the continuum without lines nor contribution of them
std0 = np.where(l>input('lim inf for determining the stddev of the continuum (angs)?: '))[0][0]
std1 = np.where(l<input('lim sup for determining the stddev of the continuum (angs)?: '))[0][-1]
stadev = np.std(data_cor[std0:std1])

###################################### Start the fit and the MODEL ###############################################
#
# First we have to initialise the model in the SII lines by doing
lin_mod = lmfit.Model(linear)
sII_mod = lmfit.Model(twogaussian)
twosII_mod = lmfit.Model(funcSII2comp)
# and initialise the model in the whole spectra for several different models
comp_mod = lmfit.Model(funcgauslin)
broad_mod = lmfit.Model(funcbroad)
twocomp_mod = lmfit.Model(func2com)

# We make the linear fit only with some windows of the spectra, and calculate the line to introduce it in the formula
linresu  = lin_mod.fit(newflux,slope=in_slope,intc=in_intc,x=newl)
new_slop = linresu.values['slope']
new_intc = linresu.values['intc']

# Now we define the initial guesses and the constraints
paramsSII = lmfit.Parameters()
params2SII = lmfit.Parameters()
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
print('The method to be applied is S-method as there are not OI lines available!')	# Method to fit
cd = lmfit.Parameter('mu_0', value=mu0)
de = lmfit.Parameter('sig_0', value=sig0)
ef = lmfit.Parameter('amp_0', value=amp0)
fg = lmfit.Parameter('mu_1', value=mu1,expr='mu_0*(6716./6731.)')
gh = lmfit.Parameter('sig_1', value=sig1,expr='sig_0')
hi = lmfit.Parameter('amp_1', value=amp1)

# second components
aaa = lmfit.Parameter('mu_20', value=mu0)
aab = lmfit.Parameter('sig_20', value=sig20)
aac = lmfit.Parameter('amp_20', value=amp20)
aad = lmfit.Parameter('mu_21', value=mu1,expr='mu_20*(6716./6731.)')
aae = lmfit.Parameter('sig_21', value=sig21,expr='sig_20')
aaf = lmfit.Parameter('amp_21', value=amp21)

# add a sequence of Parameters
paramsSII.add_many(cd,de,ef,fg,gh,hi)
params2SII.add_many(cd,de,ef,fg,gh,hi,aaa,aab,aac,aad,aae,aaf)

###################################################################################################################
# and make the fit using lmfit
SIIresu = sII_mod.fit(data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20],paramsSII,x=l[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20])
twoSIIresu = twosII_mod.fit(data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20],params2SII,x=l[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20])

##################################### PLOT and PRINT for the SII lines ##################################################
#
# Now we create the individual gaussians in order to plot and print the results for only 1 component
print('				RESULTS OF THE FIT: ')
print('Linear fit equation: {:.5f}*x + {:.5f}'.format(linresu.values['slope'], linresu.values['intc']))
print('')
print('The rest of the results can be displayed all together with two/SIIresu.params; the data can be accesed with two/SIIresu.values['']')
print('1 gaussian for SII '+str(SIIresu.params['mu_0']))
print('1 gaussian for SII '+str(SIIresu.params['sig_0']))
print('1 gaussian for SII '+str(SIIresu.params['amp_0']))
print('1 gaussian for SII '+str(SIIresu.params['mu_1']))
print('1 gaussian for SII '+str(SIIresu.params['sig_1']))
print('1 gaussian for SII '+str(SIIresu.params['amp_1']))
print('2 gaussian for SII '+str(twoSIIresu.params['mu_0']))
print('2 gaussian for SII '+str(twoSIIresu.params['sig_0']))
print('2 gaussian for SII '+str(twoSIIresu.params['amp_0']))
print('2 gaussian for SII '+str(twoSIIresu.params['mu_1']))
print('2 gaussian for SII '+str(twoSIIresu.params['sig_1']))
print('2 gaussian for SII '+str(twoSIIresu.params['amp_1']))
print('2 gaussian for SII '+str(twoSIIresu.params['mu_20']))
print('2 gaussian for SII '+str(twoSIIresu.params['sig_20']))
print('2 gaussian for SII '+str(twoSIIresu.params['amp_20']))
print('2 gaussian for SII '+str(twoSIIresu.params['mu_21']))
print('2 gaussian for SII '+str(twoSIIresu.params['sig_21']))
print('2 gaussian for SII '+str(twoSIIresu.params['amp_21']))
print('')
print('The chi-square of the fit for 1 gaussian for SII is: {:.5f}'.format(SIIresu.chisqr))
print('The chi-square of the fit for 2 gaussian for SII is: {:.5f}'.format(twoSIIresu.chisqr))
print('')

# Now we create and plot the individual gaussians of the fit
gaus1 = gaussian(l,SIIresu.values['mu_0'],SIIresu.values['sig_0'],SIIresu.values['amp_0']) 
gaus2 = gaussian(l,SIIresu.values['mu_1'],SIIresu.values['sig_1'],SIIresu.values['amp_1'])
gaus21 = gaussian(l,twoSIIresu.values['mu_0'],twoSIIresu.values['sig_0'],twoSIIresu.values['amp_0']) 
gaus22 = gaussian(l,twoSIIresu.values['mu_1'],twoSIIresu.values['sig_1'],twoSIIresu.values['amp_1'])
gaus23 = gaussian(l,twoSIIresu.values['mu_20'],twoSIIresu.values['sig_20'],twoSIIresu.values['amp_20'])
gaus24 = gaussian(l,twoSIIresu.values['mu_21'],twoSIIresu.values['sig_21'],twoSIIresu.values['amp_21'])
SIIfin_fit = twogaussian(l,SIIresu.values['mu_0'],SIIresu.values['sig_0'],SIIresu.values['amp_0'],
			 SIIresu.values['mu_1'],SIIresu.values['sig_1'],SIIresu.values['amp_1'])
SII2fin_fit = funcSII2comp(l,twoSIIresu.values['mu_0'],twoSIIresu.values['sig_0'],twoSIIresu.values['amp_0'],
			   twoSIIresu.values['mu_1'],twoSIIresu.values['sig_1'],twoSIIresu.values['amp_1'],
			   twoSIIresu.values['mu_20'],twoSIIresu.values['sig_20'],twoSIIresu.values['amp_20'],
			   twoSIIresu.values['mu_21'],twoSIIresu.values['sig_21'],twoSIIresu.values['amp_21'])

# one component
std_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]]-SIIfin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
std_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]]-SIIfin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
print('		'+str(std_s2)+'< '+str(3*stadev))
print('		'+str(std_s1)+'< '+str(3*stadev))
# two components
std2_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]]-SII2fin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
std2_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]]-SII2fin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
print('		'+str(std2_s2)+'< '+str(3*stadev))
print('		'+str(std2_s1)+'< '+str(3*stadev))

# We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
maxS1 = max(SIIfin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
maxS2 = max(SIIfin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
max2S1 = max(SII2fin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
max2S2 = max(SII2fin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
# one component
vS2 = v_luz*((SIIresu.values['mu_0']-l_SII_2)/l_SII_2)
evS2 = (v_luz/l_SII_2)*SIIresu.params['mu_0'].stderr
sigS2 = 47*np.sqrt(SIIresu.values['sig_0']**2-sig_inst**2)
esigS2 = 47*np.sqrt(SIIresu.values['sig_0']*SIIresu.params['sig_0'].stderr)/(np.sqrt(SIIresu.values['sig_0']**2-sig_inst**2))
# two comps
v2S2 = v_luz*((twoSIIresu.values['mu_0']-l_SII_2)/l_SII_2)
v20S2 = v_luz*((twoSIIresu.values['mu_20']-l_SII_2)/l_SII_2)
ev2S2 = (v_luz/l_SII_2)*twoSIIresu.params['mu_0'].stderr
ev20S2 = (v_luz/l_SII_2)*twoSIIresu.params['mu_20'].stderr
sig2S2 = 47*np.sqrt(twoSIIresu.values['sig_0']**2-sig_inst**2)
sig20S2 = 47*np.sqrt(twoSIIresu.values['sig_20']**2-sig_inst**2)
esig2S2 = 47*np.sqrt(twoSIIresu.values['sig_0']*twoSIIresu.params['sig_0'].stderr)/(np.sqrt(twoSIIresu.values['sig_0']**2-sig_inst**2))
esig20S2 = 47*np.sqrt(twoSIIresu.values['sig_20']*twoSIIresu.params['sig_20'].stderr)/(np.sqrt(twoSIIresu.values['sig_20']**2-sig_inst**2))

################################################ PLOT ######################################################
plt.close()
# MAIN plot
fig1   = plt.figure(1,figsize=(10, 8))
frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,SIIfin_fit,'r--')
plt.plot(l,gaus1,'c--')
plt.plot(l,gaus2,'c--',label='N')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
		    r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(maxS2)+' $10^{-14}$',
		    r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxS1)+' $10^{-14}$'))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
frame1.text(6850.,SIIresu.values['amp_0']+12., textstr, fontsize=12,verticalalignment='top', bbox=props)
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.plot(l,data_cor-SIIfin_fit,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
plt.ylim(-2,2)

plt.savefig(path+'adj_metS_SII_1comp.png')

#######################################################################################
# Two components in SII
# MAIN plot
fig2   = plt.figure(2,figsize=(10, 8))
frame3 = fig2.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,SII2fin_fit,'r--')
plt.plot(l,gaus21,'c--')
plt.plot(l,gaus22,'c--',label='N')
plt.plot(l,gaus23,'m--')
plt.plot(l,gaus24,'m--',label='S')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
textstr = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2S2,ev2S2),
		    r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v20S2,ev20S2),
		    r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2S2,esig2S2),
		    r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig20S2,esig20S2),
		    r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(max2S2)+' $10^{-14}$',
		    r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(max2S1)+' $10^{-14}$'))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
frame3.text(6850.,twoSIIresu.values['amp_0']+12., textstr, fontsize=12,verticalalignment='top', bbox=props)
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

frame3.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame4 = fig2.add_axes((.1,.1,.8,.2))
plt.plot(l,data_cor-SII2fin_fit,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
plt.ylim(-2,2)

plt.savefig(path+'adj_metS_SII_2comp.png')

##############################################################################################################################################################################
# We make an F-test to see if it is significant the presence of a second component in the lines. 
# As the only possible method here is the S-method due to the fact that there are no O-lines in this spectra, 
# then the method can only be applied to the SII lines (so the wavelength range would be around this two lines)

fvalue, pvalue = stats.f_oneway(SIIresu.residual,twoSIIresu.residual)

print('')
print('The probability of a second component (one component vs two components) in this spectra is: '+str(pvalue))
print('')


#######################################################################################################################################
# Select if one or two components in the SII lines and then apply to the rest
trigger = input('Is the fit good enough with one component? (Y/N): ')

# Initial guesses of the fitting parameters
sig2 = 1.				# NII2
mu2 = newx3[np.argmax(newy3)]
amp2 = max(newy3)
sig3 = 1.				# Halpha
mu3 = newx4[np.argmax(newy4)]
amp3 = max(newy4)
sig4 = 1.				# NII1
mu4 = newx5[np.argmax(newy5)]
amp4 = max(newy5)


if trigger == 'Y':
    # Now we define the initial guesses and the constraints
    params = lmfit.Parameters()
    cd = lmfit.Parameter('mu_0', value=SIIresu.values["mu_0"],vary=False)
    de = lmfit.Parameter('sig_0', value=SIIresu.values["sig_0"],vary=False)
    ef = lmfit.Parameter('amp_0', value=SIIresu.values["amp_0"],vary=False)
    fg = lmfit.Parameter('mu_1', value=SIIresu.values["mu_1"],vary=False)
    gh = lmfit.Parameter('sig_1', value=SIIresu.values["sig_1"],vary=False)
    hi = lmfit.Parameter('amp_1', value=SIIresu.values["amp_1"],vary=False)
    ij = lmfit.Parameter('mu_2', value=mu2,expr='mu_0*(6584./6731.)')
    jk = lmfit.Parameter('sig_2', value=sig2,expr='sig_0')
    kl = lmfit.Parameter('amp_2', value=amp2)
    lm = lmfit.Parameter('mu_3', value=mu3,expr='mu_0*(6563./6731.)')
    mn = lmfit.Parameter('sig_3', value=sig3,expr='sig_0')
    no = lmfit.Parameter('amp_3', value=amp3)
    op = lmfit.Parameter('mu_4', value=mu4,expr='mu_0*(6548./6731.)')
    pq = lmfit.Parameter('sig_4', value=sig4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value=amp4,expr='amp_2*(1./3.)')

    params.add_many(cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr)
    resu1 = comp_mod.fit(data_cor,params,x=l)
    
    ################################## Calculate gaussians and final fit #######################################
    # Now we create and plot the individual gaussians of the fit
    gaus1 = gaussian(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0']) 
    gaus2 = gaussian(l,resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'])
    gaus3 = gaussian(l,resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2']) 
    gaus4 = gaussian(l,resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'])
    gaus5 = gaussian(l,resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'])
    fin_fit = funcgauslin(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0'],
			  resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'],
			  resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2'],
			  resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'],
			  resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'])

    # one component
    stdf_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]]-fin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
    stdf_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]]-fin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
    print('		'+str(stdf_s2)+'< '+str(3*stadev))
    print('		'+str(stdf_s1)+'< '+str(3*stadev))

    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    maxfS1 = max(fin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    maxfS2 = max(fin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    maxfN1 = max(fin_fit[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
    maxfHa = max(fin_fit[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
    maxfN2 = max(fin_fit[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])

    ################################################ PLOT ######################################################
    plt.close('all')
    # MAIN plot
    fig1   = plt.figure(1,figsize=(10, 8))
    frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(l,data_cor)			     # Initial data
    plt.plot(l,fin_fit,'r--')
    plt.plot(l,gaus1,'c--')
    plt.plot(l,gaus2,'c--')
    plt.plot(l,gaus3,'c--')
    plt.plot(l,gaus4,'c--')
    plt.plot(l,gaus5,'c--',label='N')
    plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
    textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
		    r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(maxfS2)+' $10^{-14}$',
		    r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxfS1)+' $10^{-14}$',
		    r'$F_{NII_{2}}$ = '+ '{:.3f}'.format(maxfN2)+' $10^{-14}$',
		    r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxfHa)+' $10^{-14}$',
		    r'$F_{NII_{1}}$ = '+ '{:.3f}'.format(maxfN1)+' $10^{-14}$'))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    frame1.text(6850.,SIIresu.values['amp_0']+12., textstr, fontsize=12,verticalalignment='top', bbox=props)
    plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

    frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
    plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(l[0],l[-1])
    plt.legend(loc='best')

    # RESIDUAL plot
    frame2 = fig1.add_axes((.1,.1,.8,.2))
    plt.plot(l,data_cor-fin_fit,color='grey')		# Main
    plt.xlabel('Wavelength ($\AA$)',fontsize=14)
    plt.ylabel('Residuals',fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(l[0],l[-1])
    plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
    plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
    plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
    plt.ylim(-2,2)

    plt.savefig(path+'adj_metS_full_1comp.png')
    
    ########################################################################################################################33    
    
    trigger2 = input('Do the fit needs a broad Halpha component? (Y/N): ')
    if trigger2 == 'N': 
	print('The final plots are already printed and have been already saved!')
    elif trigger2 == 'Y':
        # Now we define the initial guesses and the constraints
	newxb = l[np.where(l>l9)[0][0]:np.where(l<l6)[0][-1]]
	newyb = data_cor[np.where(l>l9)[0][0]:np.where(l<l6)[0][-1]]
	sigb = 16.
	mub  = mu3
	ampb = amp3/3.
	paramsbH = lmfit.Parameters()
	# broad components
	ab = lmfit.Parameter('mu_b',value=mub)
	bc = lmfit.Parameter('sig_b',value=sigb)
	rs = lmfit.Parameter('amp_b',value=ampb)
	paramsbH.add_many(ab,bc,rs,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr)

    	broadresu = broad_mod.fit(data_cor,paramsbH,x=l)

        ##################################### PLOT and PRINT ##################################################
	
	
'''    
elif trigger == 'N':
    sig22 = 10.				# NII2
    amp22 = max(newy3)/2.
    sig23 = 10.				# Halpha
    amp23 = max(newy4)/2.
    sig24 = 10.				# NII1
    amp24 = max(newy5)/2.

    # Now we define the initial guesses and the constraints
    params2c = lmfit.Parameters()
    cd = lmfit.Parameter('mu_0', value=mu0,expr='twoSIIresu.values['mu_0']')
    de = lmfit.Parameter('sig_0', value=sig0,expr='twoSIIresu.values['sig_0']')
    ef = lmfit.Parameter('amp_0', value=amp0,expr='twoSIIresu.values['amp_0']')
    fg = lmfit.Parameter('mu_1', value=mu1,expr='twoSIIresu.values['mu_1'])
    gh = lmfit.Parameter('sig_1', value=sig1,expr='twoSIIresu.values['sig_1'])
    hi = lmfit.Parameter('amp_1', value=amp1,expr='twoSIIresu.values['amp_1']')
    ij = lmfit.Parameter('mu_2', value=mu2,expr='twoSIIresu.values['mu_0']*(6584./6731.)')
    jk = lmfit.Parameter('sig_2', value=sig2,expr='twoSIIresu.values['sig_0']')
    kl = lmfit.Parameter('amp_2', value=amp2)
    lm = lmfit.Parameter('mu_3', value=mu3,expr='mu_0*(6563./6731.)')
    mn = lmfit.Parameter('sig_3', value=sig3,expr='sig_0')
    no = lmfit.Parameter('amp_3', value=amp3)
    op = lmfit.Parameter('mu_4', value=mu4,expr='mu_0*(6548./6731.)')
    pq = lmfit.Parameter('sig_4', value=sig4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value=amp4,expr='amp_2*(1./3.)')
    # second components
    aaa = lmfit.Parameter('mu_20', value=mu0,expr='twoSIIresu.values['mu_20']')
    aab = lmfit.Parameter('sig_20', value=sig20,expr='twoSIIresu.values['sig_20']')
    aac = lmfit.Parameter('amp_20', value=amp20,expr='twoSIIresu.values['amp_20']')
    aad = lmfit.Parameter('mu_21', value=mu1,expr='twoSIIresu.values['mu_21']')
    aae = lmfit.Parameter('sig_21', value=sig21,expr='twoSIIresu.values['sig_21']')
    aaf = lmfit.Parameter('amp_21', value=amp21,expr='twoSIIresu.values['amp_21']')
    aag = lmfit.Parameter('mu_22', value=mu2,expr='mu_20*(6584./6731.)')
    aah = lmfit.Parameter('sig_22', value=sig22,expr='sig_20')
    aai = lmfit.Parameter('amp_22', value=amp22)
    aaj = lmfit.Parameter('mu_23', value=mu3,expr='mu_20*(6563./6731.)')
    aak = lmfit.Parameter('sig_23', value=sig23,expr='sig_20')
    aal = lmfit.Parameter('amp_23', value=amp23,min=0.)
    aam = lmfit.Parameter('mu_24', value=mu4,expr='mu_20*(6548./6731.)')
    aan = lmfit.Parameter('sig_24', value=sig24,expr='sig_20')
    aao = lmfit.Parameter('amp_24', value=amp24,expr='amp_22*(1./3.)')
    params2c.add_many(cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,aaa,aab,aac,aad,aae,aaf,aag,aah,aai,aaj,aak,aal,aam,aan,aao)

    twocompresu = twocomp_mod.fit(data_cor,params2c,x=l)

    ##################################### PLOT and PRINT ##################################################
    ######################################### PLOT and PRINT #######################################################
    # Now we create and plot the individual gaussians of the fit
    gaus1 = gaussian(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0']) 
    gaus2 = gaussian(l,resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'])
    gaus21 = gaussian(l,twoSIIresu.values['mu_0'],twoSIIresu.values['sig_0'],twoSIIresu.values['amp_0']) 
    gaus22 = gaussian(l,twoSIIresu.values['mu_1'],twoSIIresu.values['sig_1'],twoSIIresu.values['amp_1'])
    gaus23 = gaussian(l,twoSIIresu.values['mu_20'],twoSIIresu.values['sig_20'],twoSIIresu.values['amp_20'])
    gaus24 = gaussian(l,twoSIIresu.values['mu_21'],twoSIIresu.values['sig_21'],twoSIIresu.values['amp_21'])
    SIIfin_fit = twogaussian(l,SIIresu.values['mu_0'],SIIresu.values['sig_0'],SIIresu.values['amp_0'],
			 SIIresu.values['mu_1'],SIIresu.values['sig_1'],SIIresu.values['amp_1'])
    SII2fin_fit = funcSII2comp(l,twoSIIresu.values['mu_0'],twoSIIresu.values['sig_0'],twoSIIresu.values['amp_0'],
			   twoSIIresu.values['mu_1'],twoSIIresu.values['sig_1'],twoSIIresu.values['amp_1'],
			   twoSIIresu.values['mu_20'],twoSIIresu.values['sig_20'],twoSIIresu.values['amp_20'],
			   twoSIIresu.values['mu_21'],twoSIIresu.values['sig_21'],twoSIIresu.values['amp_21'])

    # one component
    std_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]]-SIIfin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
    std_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]]-SIIfin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
    print('		'+str(std_s2)+'< '+str(3*stadev))
    print('		'+str(std_s1)+'< '+str(3*stadev))
    # two components
    std2_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]]-SII2fin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
    std2_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]]-SII2fin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
    print('		'+str(std2_s2)+'< '+str(3*stadev))
    print('		'+str(std2_s1)+'< '+str(3*stadev))

    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    maxS1 = max(SIIfin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    maxS2 = max(SIIfin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    max2S1 = max(SII2fin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    max2S2 = max(SII2fin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    # one component
    vS2 = v_luz*((SIIresu.values['mu_0']-l_SII_2)/l_SII_2)
    evS2 = (v_luz/l_SII_2)*SIIresu.params['mu_0'].stderr
    sigS2 = 47*np.sqrt(SIIresu.values['sig_0']**2-sig_inst**2)
    esigS2 = 47*np.sqrt(SIIresu.values['sig_0']*SIIresu.params['sig_0'].stderr)/(np.sqrt(SIIresu.values['sig_0']**2-sig_inst**2))
    # two comps
    v2S2 = v_luz*((twoSIIresu.values['mu_0']-l_SII_2)/l_SII_2)
    v20S2 = v_luz*((twoSIIresu.values['mu_20']-l_SII_2)/l_SII_2)
    ev2S2 = (v_luz/l_SII_2)*twoSIIresu.params['mu_0'].stderr
    ev20S2 = (v_luz/l_SII_2)*twoSIIresu.params['mu_20'].stderr
    sig2S2 = 47*np.sqrt(twoSIIresu.values['sig_0']**2-sig_inst**2)
    sig20S2 = 47*np.sqrt(twoSIIresu.values['sig_20']**2-sig_inst**2)
    esig2S2 = 47*np.sqrt(twoSIIresu.values['sig_0']*twoSIIresu.params['sig_0'].stderr)/(np.sqrt(twoSIIresu.values['sig_0']**2-sig_inst**2))
    esig20S2 = 47*np.sqrt(twoSIIresu.values['sig_20']*twoSIIresu.params['sig_20'].stderr)/(np.sqrt(twoSIIresu.values['sig_20']**2-sig_inst**2))

    ################################################ PLOT ######################################################
    plt.close()
    # MAIN plot
    fig1   = plt.figure(1,figsize=(10, 8))
    frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(l,data_cor)			     # Initial data
    plt.plot(l,SIIfin_fit,'r--')
    plt.plot(l,gaus1,'c--')
    plt.plot(l,gaus2,'c--',label='N')
    plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
    textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
		    r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(maxS2)+' $10^{-14}$',
		    r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxS1)+' $10^{-14}$'))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    frame1.text(6850.,SIIresu.values['amp_0']+12., textstr, fontsize=12,verticalalignment='top', bbox=props)
    plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

    frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
    plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(l[0],l[-1])
    plt.legend(loc='best')

    # RESIDUAL plot
    frame2 = fig1.add_axes((.1,.1,.8,.2))
    plt.plot(l,data_cor-SIIfin_fit,color='grey')		# Main
    plt.xlabel('Wavelength ($\AA$)',fontsize=14)
    plt.ylabel('Residuals',fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(l[0],l[-1])
    plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
    plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
    plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
    plt.ylim(-2,2)

    plt.savefig(path+'adj_metS_SII_1comp.png')


else: 
    print('Please use "Y" or "N"')
'''

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

'''

#resu  = sing_mod.fit(data_cor,mu=mu3,sigm=sig3,amp=amp3,x=l)



#
# For the lines the stddev should be calculates:
#     - In SII/OI for the S/O methods.
#     - In Halpha/NII in order to look for a broad Halpha component.
#     - In both [OI] and [SII] with the mix models.
#
#liminf   = np.where(l>input('lim inf for determining the stddev of the line 1 residual (angs)?: '))[0][0]	# Strongest SII line
#limsup   = np.where(l<input('lim inf for determining the stddev of the line 1 residual (angs)?: '))[0][-1]
#std_line = np.std(resu1.residual[liminf:limsup])

# As we are yet testing, let's calculate it for all the lines
# one component
std_s2 = np.std(resu1.residual[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
std_s1 = np.std(resu1.residual[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
std_n2 = np.std(resu1.residual[np.where(l<l5)[0][-1]:np.where(l>l6)[0][-1]])
std_ha = np.std(resu1.residual[np.where(l<l7)[0][-1]:np.where(l>l8)[0][-1]])
std_n1 = np.std(resu1.residual[np.where(l<l9)[0][-1]:np.where(l>l10)[0][-1]])
# one component + broad halpha
stdb_s2 = np.std(broadresu.residual[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
stdb_s1 = np.std(broadresu.residual[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
stdb_n2 = np.std(broadresu.residual[np.where(l<l5)[0][-1]:np.where(l>l6)[0][-1]])
stdb_ha = np.std(broadresu.residual[np.where(l<l7)[0][-1]:np.where(l>l8)[0][-1]])
stdb_n1 = np.std(broadresu.residual[np.where(l<l9)[0][-1]:np.where(l>l10)[0][-1]])
# two components
std2_s2 = np.std(twocompresu.residual[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
std2_s1 = np.std(twocompresu.residual[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
std2_n2 = np.std(twocompresu.residual[np.where(l<l5)[0][-1]:np.where(l>l6)[0][-1]])
std2_ha = np.std(twocompresu.residual[np.where(l<l7)[0][-1]:np.where(l>l8)[0][-1]])
std2_n1 = np.std(twocompresu.residual[np.where(l<l9)[0][-1]:np.where(l>l10)[0][-1]])


####################################################################################################################
########################################## V calculus and F-test ###################################################
####################################################################################################################
#
# In order to calculate the velocity of the lines, we have to determine the redshift and then apply it to 
# follow the formula: v = cz. The error will be = c/lambda * error_lambda

v_SII2 = v_luz*((resu1.values['mu_0']-l_SII_2)/l_SII_2)
v_SII1 = v_luz*((resu1.values['mu_1']-l_SII_1)/l_SII_1)
v_NII2 = v_luz*((resu1.values['mu_2']-l_NII_2)/l_NII_2)
v_Halpha = v_luz*((resu1.values['mu_3']-l_Halpha)/l_Halpha)
v_NII1 = v_luz*((resu1.values['mu_4']-l_NII_1)/l_NII_1)
erv_SII2 = (v_luz/l_SII_2)*resu1.params['mu_0'].stderr
erv_SII1 = (v_luz/l_SII_1)*resu1.params['mu_1'].stderr
erv_NII2 = (v_luz/l_NII_2)*resu1.params['mu_2'].stderr
erv_Halpha = (v_luz/l_Halpha)*resu1.params['mu_3'].stderr
erv_NII1 = (v_luz/l_NII_1)*resu1.params['mu_4'].stderr


# We make an F-test to see if it is significant the presence of a second component in the lines. 
# As the only possible method here is the S-method due to the fact that there are no O-lines in this spectra, 
# then the method can only be applied to the SII lines (so the wavelength range would be around this two lines)

fvalue, pvalue = stats.f_oneway(resu1.residual[np.where(l<l3)[0][-1]-10:np.where(l>l2)[0][-1]+10],broadresu.residual[np.where(l<l3)[0][-1]-10:np.where(l>l2)[0][-1]+10])
fvalue2, pvalue2 = stats.f_oneway(resu1.residual[np.where(l<l3)[0][-1]-10:np.where(l>l2)[0][-1]+10],twocompresu.residual[np.where(l<l3)[0][-1]-10:np.where(l>l2)[0][-1]+10])

print('')
print('The probability of a second component (one component vs one + broad Halpha) in this spectra is: '+str(pvalue))
print('The probability of a second component (one component vs two components) in this spectra is: '+str(pvalue2))
print('')

########################################################################################################################
########################################################################################################################
############################################## RESULTS: PLOT and PRINT #################################################
########################################################################################################################
########################################################################################################################
#
# Now we create the individual gaussians in order to plot and print the results for only 1 component
print('				RESULTS OF THE FIT: ')
print('Linear fit equation: {:.5f}*x + {:.5f}'.format(linresu.values['slope'], linresu.values['intc']))
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
#print('The standard deviation of the continuum is: {:.5f}  and the one of the SII line is: {:.5f}'.format(stadev, std_line))
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> ')
print('		str(std_s2)+'< 3*'+str(stadev)')
print('		str(std_s1)+'< 3*'+str(stadev)')
print('		str(std_n2)+'< 3*'+str(stadev)')
print('		str(std_ha)+'< 3*'+str(stadev)')
print('		str(std_n1)+'< 3*'+str(stadev)')

# Now we create and plot the individual gaussians of the fit
gaus1 = gaussian(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0']) 
gaus2 = gaussian(l,resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'])
gaus3 = gaussian(l,resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2'])
gaus4 = gaussian(l,resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'])
gaus5 = gaussian(l,resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'])
final_fit = funcgauslin(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0'],
		       resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'],
		       resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2'],
		       resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'],
		       resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'])

# We determine the maximum flux of the fit for all the lines
maxN1 = max(final_fit[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
maxHa = max(final_fit[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
maxN2 = max(final_fit[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])
maxS1 = max(final_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
maxS2 = max(final_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])


##########################################################################################################################
############################################# PRINT AND PLOT 1 COMP + HA #################################################
#
# Now we create the individual gaussians in order to plot and print the results for only 1 component
print('			RESULTS OF THE FIT  WITH A BROAD COMPONENT: ')
print('Linear fit equation: {:.5f}*x + {:.5f}'.format(linresu.values['slope'], linresu.values['intc']))
print('')
print('The rest of the results can be displayed all together with broadresu.params; the data can be accesed with broadresu.values['']')
print(broadresu.params['mu_0'])
print(broadresu.params['sig_0'])
print(broadresu.params['amp_0'])
print(broadresu.params['mu_1'])
print(broadresu.params['sig_1'])
print(broadresu.params['amp_1'])
print(broadresu.params['mu_2'])
print(broadresu.params['sig_2'])
print(broadresu.params['amp_2'])
print(broadresu.params['mu_3'])
print(broadresu.params['sig_3'])
print(broadresu.params['amp_3'])
print(broadresu.params['mu_4'])
print(broadresu.params['sig_4'])
print(broadresu.params['amp_4'])
print(broadresu.params['mu_b'])
print(broadresu.params['sig_b'])
print(broadresu.params['amp_b'])
print('')
print('The chi-square of the fit is: {:.5f}'.format(broadresu.chisqr))
print('')
#print('The standard deviation of the continuum is: {:.5f}  and the one of the SII line is: {:.5f}'.format(stadev, std_line))
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> ')
print('		str(stdb_s2)+'< 3*'+str(stadev)')
print('		str(stdb_s1)+'< 3*'+str(stadev)')
print('		str(stdb_n2)+'< 3*'+str(stadev)')
print('		str(stdb_ha)+'< 3*'+str(stadev)')
print('		str(stdb_n1)+'< 3*'+str(stadev)')

# Now we create and plot the individual gaussians of the fit
bgaus1 = gaussian(l,broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0']) 
bgaus2 = gaussian(l,broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'])
bgaus3 = gaussian(l,broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2'])
bgaus4 = gaussian(l,broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'])
bgaus5 = gaussian(l,broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'])
bgaus6 = gaussian(l,broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])

# We determine the maximum flux of the fit for all the lines
finalb_fit = funcbroad(l,broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0'],
		       broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'],
		       broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2'],
		       broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'],
		       broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'],
		       broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])
maxbN1 = max(finalb_fit[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
maxbHa = max(finalb_fit[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
maxbN2 = max(finalb_fit[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])
maxbS1 = max(finalb_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
maxbS2 = max(finalb_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])

################################################ Broad Component plot ###################################################
plt.figure()
# MAIN plot
fig2   = plt.figure(2)
frame2 = fig2.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,finalb_fit,'r--')
plt.plot(l,bgaus1,'c--')
plt.plot(l,bgaus2,'c--')
plt.plot(l,bgaus3,'c--')
plt.plot(l,bgaus4,'c--')
plt.plot(l,bgaus5,'c--',label='N')
plt.plot(l,bgaus6,'m--',label='B')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
t=plt.text(6900.,broadresu.values['amp_2']+2.4,r'$V_{SII_{2}}$ = '+ '{:.3f}'.format(v_luz*((broadresu.values['mu_0']-l_SII_2)/l_SII_2))+' km/s',size='large')	# print v
t=plt.text(6900.,broadresu.values['amp_2']+1.6,r'$\sigma_{SII_{2}}$ = '+ '{:.3f}'.format(np.sqrt(broadresu.values['sig_0']**2-sig_inst**2)),size='large')	# print sigma already in km/s
t=plt.text(6900.,broadresu.values['amp_2']+0.8,r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(maxbS2)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,broadresu.values['amp_2'],r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxbS1)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,broadresu.values['amp_2']-0.8,r'$F_{NII_{2}}$ = '+ '{:.3f}'.format(maxbN2)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,broadresu.values['amp_2']-1.6,r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxbHa)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,broadresu.values['amp_2']-2.4,r'$F_{NII_{1}}$ = '+ '{:.3f}'.format(maxbN1)+' $10^{-14}$',size='large')	# print sigma
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

# 3-sigma of the fit --> evaluate the uncertainty in the model with a specified level for sigma
#dely = broadresu.eval_uncertainty(sigma=3)
#plt.fill_between(l,broadresu.best_fit-dely,broadresu.best_fit+dely, color="wheat")
dely = linresu.eval_uncertainty(sigma=3)
plt.fill_between(newl,linresu.best_fit-dely,linresu.best_fit+dely, color="wheat")

frame2.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame3 = fig2.add_axes((.1,.1,.8,.2))
plt.plot(l,data_cor-finalb_fit,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit

plt.savefig(path+'adj_metS_1comp_broadH.png')

##########################################################################################################################
############################################# PRINT AND PLOT 2 COMP ######################################################
##########################################################################################################################
#
# Now we create the individual gaussians in order to plot and print the results for only 1 component
print('			RESULTS OF THE FIT WITH TWO COMPONENT: ')
print('Linear fit equation: {:.5f}*x + {:.5f}'.format(linresu.values['slope'], linresu.values['intc']))
print('')
print('The rest of the results can be displayed all together with twocompresu.params; the data can be accesed with twocompresu.values['']')
print(twocompresu.params['mu_0'])
print(twocompresu.params['sig_0'])
print(twocompresu.params['amp_0'])
print(twocompresu.params['mu_1'])
print(twocompresu.params['sig_1'])
print(twocompresu.params['amp_1'])
print(twocompresu.params['mu_2'])
print(twocompresu.params['sig_2'])
print(twocompresu.params['amp_2'])
print(twocompresu.params['mu_3'])
print(twocompresu.params['sig_3'])
print(twocompresu.params['amp_3'])
print(twocompresu.params['mu_4'])
print(twocompresu.params['sig_4'])
print(twocompresu.params['amp_4'])
print(twocompresu.params['mu_20'])
print(twocompresu.params['sig_20'])
print(twocompresu.params['amp_20'])
print(twocompresu.params['mu_21'])
print(twocompresu.params['sig_21'])
print(twocompresu.params['amp_21'])
print(twocompresu.params['mu_22'])
print(twocompresu.params['sig_22'])
print(twocompresu.params['amp_22'])
print(twocompresu.params['mu_23'])
print(twocompresu.params['sig_23'])
print(twocompresu.params['amp_23'])
print(twocompresu.params['mu_24'])
print(twocompresu.params['sig_24'])
print(twocompresu.params['amp_24'])
print('')
print('The chi-square of the fit is: {:.5f}'.format(twocompresu.chisqr))
print('')
#print('The standard deviation of the continuum is: {:.5f}  and the one of the SII line is: {:.5f}'.format(stadev, std_line))
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> ')
print('		str(stdb_s2)+'< 3*'+str(stadev)')
print('		str(stdb_s1)+'< 3*'+str(stadev)')
print('		str(stdb_n2)+'< 3*'+str(stadev)')
print('		str(stdb_ha)+'< 3*'+str(stadev)')
print('		str(stdb_n1)+'< 3*'+str(stadev)')

# Now we create and plot the individual gaussians of the fit
tgaus1 = gaussian(l,twocompresu.values['mu_0'],twocompresu.values['sig_0'],twocompresu.values['amp_0']) 
tgaus2 = gaussian(l,twocompresu.values['mu_1'],twocompresu.values['sig_1'],twocompresu.values['amp_1'])
tgaus3 = gaussian(l,twocompresu.values['mu_2'],twocompresu.values['sig_2'],twocompresu.values['amp_2'])
tgaus4 = gaussian(l,twocompresu.values['mu_3'],twocompresu.values['sig_3'],twocompresu.values['amp_3'])
tgaus5 = gaussian(l,twocompresu.values['mu_4'],twocompresu.values['sig_4'],twocompresu.values['amp_4'])
tgaus6 = gaussian(l,twocompresu.values['mu_20'],twocompresu.values['sig_20'],twocompresu.values['amp_20'])
tgaus7 = gaussian(l,twocompresu.values['mu_21'],twocompresu.values['sig_21'],twocompresu.values['amp_21']) 
tgaus8 = gaussian(l,twocompresu.values['mu_22'],twocompresu.values['sig_22'],twocompresu.values['amp_22'])
tgaus9 = gaussian(l,twocompresu.values['mu_23'],twocompresu.values['sig_23'],twocompresu.values['amp_23'])
tgaus10 = gaussian(l,twocompresu.values['mu_24'],twocompresu.values['sig_24'],twocompresu.values['amp_24'])

# We determine the maximum flux of the fit for all the lines
final2_fit = func2com(l,twocompresu.values['mu_0'],twocompresu.values['sig_0'],twocompresu.values['amp_0'],
		       twocompresu.values['mu_1'],twocompresu.values['sig_1'],twocompresu.values['amp_1'],
		       twocompresu.values['mu_2'],twocompresu.values['sig_2'],twocompresu.values['amp_2'],
		       twocompresu.values['mu_3'],twocompresu.values['sig_3'],twocompresu.values['amp_3'],
		       twocompresu.values['mu_4'],twocompresu.values['sig_4'],twocompresu.values['amp_4'],
		       twocompresu.values['mu_20'],twocompresu.values['sig_20'],twocompresu.values['amp_20'],
		       twocompresu.values['mu_21'],twocompresu.values['sig_21'],twocompresu.values['amp_21'],
		       twocompresu.values['mu_22'],twocompresu.values['sig_22'],twocompresu.values['amp_22'],
		       twocompresu.values['mu_23'],twocompresu.values['sig_23'],twocompresu.values['amp_23'],
		       twocompresu.values['mu_24'],twocompresu.values['sig_24'],twocompresu.values['amp_24'],)
max2N1 = max(final2_fit[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
max2Ha = max(final2_fit[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
max2N2 = max(final2_fit[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])
max2S1 = max(final2_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
max2S2 = max(final2_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
################################################ Broad Component plot ###################################################
plt.figure()
# MAIN plot
fig3   = plt.figure(3)
frame4 = fig3.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,final2_fit,'r--')
plt.plot(l,tgaus1,'c--')
plt.plot(l,tgaus2,'c--')
plt.plot(l,tgaus3,'c--')
plt.plot(l,tgaus4,'c--')
plt.plot(l,tgaus5,'c--',label='N')
plt.plot(l,tgaus6,'m--')
plt.plot(l,tgaus7,'m--')
plt.plot(l,tgaus8,'m--')
plt.plot(l,tgaus9,'m--')
plt.plot(l,tgaus10,'m--',label='S')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
t=plt.text(6900.,twocompresu.values['amp_2']+3.2,r'$V_{SII_{2-1comp}}$ = '+ '{:.3f}'.format(v_luz*((twocompresu.values['mu_0']-l_SII_2)/l_SII_2))+' km/s',size='large')		# print v
t=plt.text(6900.,twocompresu.values['amp_2']+2.4,r'$V_{SII_{2-2comp}}$ = '+ '{:.3f}'.format(v_luz*((twocompresu.values['mu_20']-l_SII_2)/l_SII_2))+' km/s',size='large')	# print v
t=plt.text(6900.,twocompresu.values['amp_2']+1.6,r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.3f}'.format(np.sqrt(twocompresu.values['sig_0']**2-sig_inst**2)),size='large')	# print sigma already in km/s
t=plt.text(6900.,twocompresu.values['amp_2']+0.8,r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.3f}'.format(np.sqrt(twocompresu.values['sig_20']**2-sig_inst**2)),size='large')	# print sigma already in km/s
t=plt.text(6900.,twocompresu.values['amp_2'],r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(max2S2)+' $10^{-14}$',size='large')		# print sigma
t=plt.text(6900.,twocompresu.values['amp_2']-0.8,r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(max2S1)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,twocompresu.values['amp_2']-1.6,r'$F_{NII_{2}}$ = '+ '{:.3f}'.format(max2N2)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,twocompresu.values['amp_2']-2.4,r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(max2Ha)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,twocompresu.values['amp_2']-3.2,r'$F_{NII_{1}}$ = '+ '{:.3f}'.format(max2N1)+' $10^{-14}$',size='large')	# print sigma
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

# 3-sigma of the fit --> evaluate the uncertainty in the model with a specified level for sigma
dely = broadresu.eval_uncertainty(sigma=3)
plt.fill_between(l,twocompresu.best_fit-dely,twocompresu.best_fit+dely, color="wheat")
dely = linresu.eval_uncertainty(sigma=3)
plt.fill_between(newl,linresu.best_fit-dely,linresu.best_fit+dely, color="wheat")

frame4.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame5 = fig3.add_axes((.1,.1,.8,.2))
plt.plot(l,data_cor-final2_fit,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit

plt.savefig(path+'adj_metS_2comp.png')
'''
