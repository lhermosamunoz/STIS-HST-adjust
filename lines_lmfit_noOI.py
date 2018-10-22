'''
This script makes a gaussian fit to the emission lines of AGN spectra
It is needed a path, the spectrum in which the fit is going to be made and the initial estimation of the fit
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lmfit
import scipy.stats as stats


######################### Define the PATHS to the data and extract the spectra ###################################
#
path      = input('Path to the data fits? (ex. "/mnt/data/lhermosa/HLA_data/NGC.../o.../ext_spec_crop.fits"): ')
hdulist   = fits.open(path)	#'/mnt/data/lhermosa/HLA_data/NGC3245/O57205030_STISCCD_G750M/ext_spec_combin_crop.fits')	# Open the fit file to read the information
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
    fy  = fy + gaussian(x,mu_0,sig_0,amp_0)
    fy  = fy + gaussian(x,mu_1,sig_1,amp_1)
    fy  = fy + gaussian(x,mu_2,sig_2,amp_2)
    fy  = fy + gaussian(x,mu_3,sig_3,amp_3)
    fy  = fy + gaussian(x,mu_4,sig_4,amp_4)
    return fy

# Function to create the second component in the fit
def func2comp(x,mu1_0,sig1_0,amp1_0,mu1_1,sig1_1,amp1_1,mu1_2,sig1_2,amp1_2,mu1_3,sig1_3,amp1_3,mu1_4,sig1_4,amp1_4):
    '''
    Function to fit the spectra to a second component.
    The parameters to introduce have to be the initial guesses. 
    The basis of the fit will be the first component fit.
    x - values for the fit
    params: 1. mu - mean of the distribution
	    2. sigma - stddev
	    3. amplitude
    '''
    fy = np.zeros_like(x)
    fy = fy + resu1.data
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

l1 = input('lambda inf for SII 2 (angs)?: ')
l2 = input('lambda sup for SII 2 (angs)?: ')
liminf = np.where(l>l1)[0][0]	# Strongest SII line 
limsup = np.where(l<l2)[0][-1]
newx1 = l[liminf:limsup+1]
newy1 = data_cor[liminf:limsup+1]
# Initial guesses of the fitting parameters
sig_0 = 1.
mu_0  = newx1[np.argmax(newy1)]
amp_0 = max(newy1)

l3 = input('lambda inf for SII 1 (angs)?: ')
l4 = input('lambda sup for SII 1 (angs)?: ')
liminf = np.where(l>l3)[0][0]
limsup = np.where(l<l4)[0][-1]
newx2 = l[liminf:limsup+1]
newy2 = data_cor[liminf:limsup+1]
sig_1 = 1.
mu_1  = newx2[np.argmax(newy2)]
amp_1 = max(newy2)

l5 = input('lambda inf for NII 2 (angs)?: ')
l6 = input('lambda sup for NII 2 (angs)?: ')
liminf = np.where(l>l5)[0][0]
limsup = np.where(l<l6)[0][-1]
newx3 = l[liminf:limsup+1]
newy3 = data_cor[liminf:limsup+1]
sig_2 = 1.
mu_2  = newx3[np.argmax(newy3)]
amp_2 = max(newy3)

l7 = input('lambda inf for Halpha (angs)?: ')
l8 = input('lambda sup for Halpha (angs)?: ')
liminf = np.where(l>l7)[0][0]
limsup = np.where(l<l8)[0][-1]
newx4 = l[liminf:limsup+1]
newy4 = data_cor[liminf:limsup+1]
sig_3 = 1.
mu_3  = newx4[np.argmax(newy4)]
amp_3 = max(newy4)

l9 = input('lambda inf for NII 1 (angs)?: ')
l10 = input('lambda sup for NII 1 (angs)?: ')
liminf = np.where(l>l9)[0][0]
limsup = np.where(l<l10)[0][-1]
newx5  = l[liminf:limsup+1]
newy5  = data_cor[liminf:limsup+1]
sig_4 = 1.
mu_4  = newx5[np.argmax(newy5)]
amp_4 = max(newy5)

#
# Start the parameters for the LINEAR fit
in_slope = 0.
in_intc  = data_cor[0]

newl = l[1]		# Redefine the lambda zone with the first and last point and the zones in between OI2-NII1 and NII2-SII1
zone_S_fin = l[np.where(l<l2)[0][-1]+10:np.where(l>7000.)[0][0]]
zone_N_S = l[np.where(l<l6)[0][-1]+10:np.where(l>l3)[0][0]-10]
newl = np.append(newl,zone_N_S)
newl = np.append(newl,zone_S_fin)
newl = np.append(newl,l[-1])
# now we do the same but with the flux data (y vector)
newflux = data_cor[1]
zon_S_fin = data_cor[np.where(l<l2)[0][-1]+10:np.where(l>7000.)[0][0]]
zon_N_S = data_cor[np.where(l<l6)[0][-1]+10:np.where(l>l3)[0][0]-10]
newflux = np.append(newflux,zon_N_S)
newflux = np.append(newflux,zon_S_fin)
newflux = np.append(newflux,data_cor[-1])


###################################### Start the fit and the MODEL ###############################################
#
# First we have to initialise the model by doing
sing_mod = lmfit.Model(gaussian)
lin_mod = lmfit.Model(linear)
comp_mod = lmfit.Model(funcgauslin)

# We make the linear fit only with some windows of the spectra, and calculate the line to introduce it in the formula
linresu  = lin_mod.fit(newflux,slope=in_slope,intc=in_intc,x=newl)
new_slop = linresu.values['slope']
new_intc = linresu.values['intc']

# Now we define the initial guesses and the constraints
params = lmfit.Parameters()
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
print('The method to be applied is S-method as there are not OI lines available!')	# Method to fit
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

# add a sequence of Parameters
params.add_many(cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr)

# and make the fit using lmfit
resu  = sing_mod.fit(data_cor,mu=mu_0,sigm=sig_0,amp=amp_0,x=l)
resu1 = comp_mod.fit(data_cor,params,x=l)

# In order to determine if the lines need one more gaussian to be fit correctly, we apply the condition
# that the std dev of the continuum should be higher than 3 times the std dev of the residuals of the 
# fit of the line. We have to calculate the stddev of the continuum in a place where there are no 
# lines (True for all AGNs spectra in this range).
# Calculate the standard deviation of a part of the continuum without lines nor contribution of them
std0 = np.where(l>input('lim inf for determining the stddev of the continuum (angs)?: '))[0][0]
std1 = np.where(l<input('lim sup for determining the stddev of the continuum (angs)?: '))[0][-1]
stadev = np.std(data_cor[std0:std1])
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
std_s2 = np.std(resu1.residual[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
std_s1 = np.std(resu1.residual[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
std_n2 = np.std(resu1.residual[np.where(l<l5)[0][-1]:np.where(l>l6)[0][-1]])
std_ha = np.std(resu1.residual[np.where(l<l7)[0][-1]:np.where(l>l8)[0][-1]])
std_n1 = np.std(resu1.residual[np.where(l<l9)[0][-1]:np.where(l>l10)[0][-1]])


###############################################################################################################
####################################### V calculus and F-test #################################################
###############################################################################################################
#
# In order to calculate the velocity of the lines, we have to determine the redshift and then apply it to 
# follow the formula: v = cz. The error will be = v/lambda * error_lambda
v_luz = 299792.458 #km/s
v_SII2 = v_luz*((resu1.values['mu_0']-l_SII_2)/l_SII_2)
v_SII1 = v_luz*((resu1.values['mu_1']-l_SII_1)/l_SII_1)
v_NII2 = v_luz*((resu1.values['mu_2']-l_NII_2)/l_NII_2)
v_Halpha = v_luz*((resu1.values['mu_3']-l_Halpha)/l_Halpha)
v_NII1 = v_luz*((resu1.values['mu_4']-l_NII_1)/l_NII_1)
erv_SII2 = (v_SII2/resu1.values['mu_0'])#*err_mu0
erv_SII1 = (v_SII1/resu1.values['mu_1'])#*err_mu1
erv_NII2 = (v_NII2/resu1.values['mu_2'])#*err_mu2
erv_Halpha = (v_Halpha/resu1.values['mu_3'])#*err_mu3
erv_NII1 = (v_NII1/resu1.values['mu_4'])#*err_mu4


# We make an F-test to see if it is significant the presence of a second component in the lines. 
# As the only possible method here is the S-method due to the fact that there are no O-lines in this spectra, 
# then the method can only be applied to the SII lines (so the wavelength range would be around this two lines)

#fvalue, pvalue = stats.f_oneway(resu1.residual[np.where(l<l3)[0][-1]-10:np.where(l>l2)[0][-1]+10],resu2.residual[np.where(l<l3)[0][-1]-10:np.where(l>l2)[0][-1]+10])
print('')
#print('The probability of a second component in this spectra is: '+str(pvalue))
print('')


###############################################################################################################
######################################### RESULTS: PLOT and PRINT #############################################
###############################################################################################################
#
# Now we create the individual gaussians in order to plot and print the results
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

# We determine the maximum flux of the fit for all the lines
maxN1 = max(resu1.data[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
maxHa = max(resu1.data[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
maxN2 = max(resu1.data[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])
maxS1 = max(resu1.data[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
maxS2 = max(resu1.data[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])

################################################ PLOT ######################################################
plt.close()
# MAIN plot
fig1   = plt.figure(1)
frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,funcgauslin(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0'],
		       resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'],
		       resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2'],
		       resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'],
		       resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4']),'r--')
plt.plot(l,gaus1,'c--')
plt.plot(l,gaus2,'c--')
plt.plot(l,gaus3,'c--')
plt.plot(l,gaus4,'c--')
plt.plot(l,gaus5,'c--',label='N')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
t=plt.text(6900.,resu1.values['amp_2']+2.4,r'$V_{SII_{2}}$ = '+ '{:.3f}'.format(v_SII2)+' km/s',size='large')			# print v
t=plt.text(6900.,resu1.values['amp_2']+1.6,r'$\sigma_{SII_{2}}$ = '+ '{:.3f}'.format(resu1.values['sig_0']),size='large')	# print sigma
t=plt.text(6900.,resu1.values['amp_2']+0.8,r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(maxS2)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,resu1.values['amp_2'],r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxS1)+' $10^{-14}$',size='large')		# print sigma
t=plt.text(6900.,resu1.values['amp_2']-0.8,r'$F_{NII_{2}}$ = '+ '{:.3f}'.format(maxN2)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,resu1.values['amp_2']-1.6,r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxHa)+' $10^{-14}$',size='large')	# print sigma
t=plt.text(6900.,resu1.values['amp_2']-2.4,r'$F_{NII_{1}}$ = '+ '{:.3f}'.format(maxN1)+' $10^{-14}$',size='large')	# print sigma
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

# 3-sigma of the fit --> evaluate the uncertainty in the model with a specified level for sigma
dely = resu1.eval_uncertainty(sigma=3)
plt.fill_between(l,resu1.best_fit-dely,resu1.best_fit+dely, color="wheat")
dely = linresu.eval_uncertainty(sigma=3)
plt.fill_between(newl,linresu.best_fit-dely,linresu.best_fit+dely, color="wheat")

frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])

# RESIDUAL plot
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.plot(l,-resu1.residual,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') # 3 sigma down limit
