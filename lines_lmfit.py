'''
This script makes a gaussian fit to the emission lines of LINERII-AGN spectra
It is needed a path to the spectrum in which the fit is going to be made & n initial estimation of the fit
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lmfit 
from lmfit.printfuncs import fit_report


######################### Define the PATHS to the data and extract the spectra ###################################
#
path = input('Path to the data fits? (ex. "/mnt/data/lhermosa/HLA_data/NGC.../o.../ext_spec_crop.fits"): ')
#hdulist = fits.open('/mnt/data/lhermosa/HLA_data/NGC3245/O57205030_STISCCD_G750M/ext_spec_combin_crop.fits')	#path)	# Open the fit file to read the information
hdulist = fits.open(path)		# Open the fit file to read the information
hdu  = hdulist[0]			# Extract the extension in which the spectra is saved
data = hdu.data			# Save the data (i.e. the values of the flux per pixel)
#hdu1 = fits.PrimaryHDU()
#hdu1.header = hdu.header
#xnew = np.arange(1,len(data)+1,1)
#hdulist.close()

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
def funcgauslin(x,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4,mu_5,sig_5,amp_5,mu_6,sig_6,amp_6):
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
    y = np.zeros_like(x)
    y = y + new_slop*x + new_intc
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_2,sig_2,amp_2) + gaussian(x,mu_3,sig_3,amp_3) + gaussian(x,mu_4,sig_4,amp_4) + gaussian(x,mu_5,sig_5,amp_5) + gaussian(x,mu_6,sig_6,amp_6)
    return y


# Broad component of Halpha
def funcbroad(x,mu_0,sig_0,amp_0,mu_1,sig_1,amp_1,mu_2,sig_2,amp_2,mu_3,sig_3,amp_3,mu_4,sig_4,amp_4,mu_5,sig_5,amp_5,mu_6,sig_6,amp_6,mu_b,sig_b,amp_b):
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
    y = y + gaussian(x,mu_0,sig_0,amp_0) + gaussian(x,mu_1,sig_1,amp_1) + gaussian(x,mu_2,sig_2,amp_2) + gaussian(x,mu_3,sig_3,amp_3) + gaussian(x,mu_4,sig_4,amp_4) + gaussian(x,mu_5,sig_5,amp_5) + gaussian(x,mu_6,sig_6,amp_6) + gaussian(x,mu_b,sig_b,amp_b)
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
cd1_1 = data_head['CD1_1']

xnew = np.arange(1,len(data)+1,1)	# We have to start in 1 for doing the transformation as no 0 pixel exist!! 
l = crval1 + (xnew-crpix1)*cd1_1	# This is the wavelength range to use. The data vector contains the flux

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
lHalpha = 6563.
lN2_1 = 6548.
lN2_2 = 6584.
lS2_1 = 6716.
lS2_2 = 6731.
lO1_1 = 6300.
lO1_2 = 6363.

# Now redefine the zone to fit
data_cor = data[2:-2]*10**14
l = l[2:-2]

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
l11 = input('lambda inf for OI 1 (angs)?: ')
l12 = input('lambda sup for OI 1 (angs)?: ')
l13 = input('lambda inf for OI 2 (angs)?: ')
l14 = input('lambda sup for OI 2 (angs)?: ')

newx1 = l[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]+1]
newy1 = data_cor[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]+1]
# Initial guesses of the fitting parameters
sig0 = 1.
mu0  = newx1[np.argmax(newy1)]
amp0 = max(newy1)

newx2 = l[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]+1]
newy2 = data_cor[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]+1]
sig1 = 1.
mu1  = newx2[np.argmax(newy2)]
amp1 = max(newy2)

newx3 = l[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]+1]
newy3 = data_cor[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]+1]
sig2 = 1.
mu2  = newx3[np.argmax(newy3)]
amp2 = max(newy3)

newx4 = l[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]+1]
newy4 = data_cor[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]+1]
sig3 = 1.
mu3  = newx4[np.argmax(newy4)]
amp3 = max(newy4)

newx5 = l[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]+1]
newy5 = data_cor[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]+1]
sig4 = 1.
mu4  = newx5[np.argmax(newy5)]
amp4 = max(newy5)

newx6 = l[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]+1]
newy6 = data_cor[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]+1]
sig5 = 1.
mu5  = newx6[np.argmax(newy6)]
amp5 = max(newy6)

newx7 = l[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]+1]
newy7 = data_cor[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]+1]
sig6 = 1.
mu6  = newx7[np.argmax(newy7)]
amp6 = max(newy7)

newxb = l[np.where(l>l9)[0][0]:np.where(l<l6)[0][-1]]
newyb = data_cor[np.where(l>l9)[0][0]:np.where(l<l6)[0][-1]]
sigb = 15.
mub = mu3
ampb = amp3/3.

#
# Start the parameters for the LINEAR fit and create the continuum zones to fit (newx)
slope = 0.
intc = data_cor[0]

newl = l[0]		# Redefine the lambda zone with the first and last point and the zones in between OI2-NII1 and NII2-SII1
zone_O_N = l[np.where(l<l14)[0][-1]+5:np.where(l>l9)[0][0]-5]
zone_N_S = l[np.where(l<l6)[0][-1]+5:np.where(l>l3)[0][0]-5]
newl = np.append(newl,zone_O_N)
newl = np.append(newl,zone_N_S)
newl = np.append(newl,l[-1])
# now we do the same but with the flux data (y vector)
newflux = data_cor[0]
zon_O_N = data_cor[np.where(l<l14)[0][-1]+5:np.where(l>l9)[0][0]-5]
zon_N_S = data_cor[np.where(l<l6)[0][-1]+5:np.where(l>l3)[0][0]-5]
newflux = np.append(newflux,zon_O_N)
newflux = np.append(newflux,zon_N_S)
newflux = np.append(newflux,data_cor[-1])


###################################### Start the fit and the MODEL ###############################################

# First we have to initialise the model by doing
sing_mod = lmfit.Model(gaussian)
lin_mod = lmfit.Model(linear)
comp_mod = lmfit.Model(funcgauslin)
broad_mod = lmfit.Model(funcbroad)

# We make the linear fit only with some windows of the spectra, and calculate the line to introduce it in the formula
linresu  = lin_mod.fit(newflux,slope=slope,intc=intc,x=newl)
new_slop = linresu.params['slope']
new_intc = linresu.params['intc']
rect = new_slop*l + new_intc

# Now we define the initial guesses and the constraints
params = lmfit.Parameters()
paramsbH = lmfit.Parameters()
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
ab = lmfit.Parameter('mu_b', value=mub)
bc = lmfit.Parameter('sig_b', value=sigb)
yz = lmfit.Parameter('amp_b', value=ampb)
meth = input('Which method do you want to use? (options are S, O, M1 or M2): ')	# Method to fit: S/O/M1/M2
if meth == 'S':
    cd = lmfit.Parameter('mu_0', value=mu0)
    de = lmfit.Parameter('sig_0', value=sig0)#,max=2.5)
    ef = lmfit.Parameter('amp_0', value=amp0)
    fg = lmfit.Parameter('mu_1', value=mu1,expr='mu_0*(6716./6731.)')
    gh = lmfit.Parameter('sig_1', value=sig1,expr='sig_0')
    hi = lmfit.Parameter('amp_1', value=amp1)
    ij = lmfit.Parameter('mu_2', value=mu2,expr='mu_0*(6583./6731.)')
    jk = lmfit.Parameter('sig_2', value=sig2,expr='sig_0')
    kl = lmfit.Parameter('amp_2', value=amp2)
    lm = lmfit.Parameter('mu_3', value=mu3,expr='mu_0*(6563./6731.)')
    mn = lmfit.Parameter('sig_3', value=sig3,expr='sig_0')
    no = lmfit.Parameter('amp_3', value=amp3)
    op = lmfit.Parameter('mu_4', value=mu4,expr='mu_0*(6548./6731.)')
    pq = lmfit.Parameter('sig_4', value=sig4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value=amp4,expr='amp_2*(1./3.)')
    rs = lmfit.Parameter('mu_5', value=mu5,expr='mu_0*(6300./6731.)')
    st = lmfit.Parameter('sig_5', value=sig5,expr='sig_0')
    tu = lmfit.Parameter('amp_5', value=amp5)
    uv = lmfit.Parameter('mu_6', value=mu6,expr='mu_0*(6363./6731.)')
    vw = lmfit.Parameter('sig_6', value=sig6,expr='sig_0')
    wy = lmfit.Parameter('amp_6', value=amp6,expr='amp_5*(1./3.)')
    params.add_many(cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,rs,st,tu,uv,vw,wy)
    paramsbH.add_many(cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,rs,st,tu,uv,vw,wy,ab,bc,yz)
elif meth == 'O':
    cd = lmfit.Parameter('mu_0', value=mu0,expr='mu_5*(6731./6300.)')
    de = lmfit.Parameter('sig_0', value=sig0,expr = 'sig_5')
    ef = lmfit.Parameter('amp_0', value=amp0)
    fg = lmfit.Parameter('mu_1', value=mu1,expr='mu_5*(6716./6300.)')
    gh = lmfit.Parameter('sig_1', value=sig1,expr='sig_5')
    hi = lmfit.Parameter('amp_1', value=amp1)
    ij = lmfit.Parameter('mu_2', value=mu2,expr='mu_5*(6583./6300.)')
    jk = lmfit.Parameter('sig_2', value=sig2,expr='sig_5')
    kl = lmfit.Parameter('amp_2', value=amp2)
    lm = lmfit.Parameter('mu_3', value=mu3,expr='mu_5*(6563./6300.)')
    mn = lmfit.Parameter('sig_3', value=sig3,expr='sig_5')
    no = lmfit.Parameter('amp_3', value=amp3)
    op = lmfit.Parameter('mu_4', value=mu4,expr='mu_5*(6548./6300.)')
    pq = lmfit.Parameter('sig_4', value=sig4,expr='sig_5')
    qr = lmfit.Parameter('amp_4', value=amp4,expr='amp_2*(1./3.)')
    rs = lmfit.Parameter('mu_5', value=mu5)
    st = lmfit.Parameter('sig_5', value=sig5)#,max=3.)
    tu = lmfit.Parameter('amp_5', value=amp5)
    uv = lmfit.Parameter('mu_6', value=mu6,expr='mu_5*(6363./6300.)')
    vw = lmfit.Parameter('sig_6', value=sig6,expr='sig_5')
    wy = lmfit.Parameter('amp_6', value=amp6,expr='amp_5*(1./3.)')

elif meth == 'M1':
    cd = lmfit.Parameter('mu_0', value=mu0)
    de = lmfit.Parameter('sig_0', value=sig0)
    ef = lmfit.Parameter('amp_0', value=amp0)
    fg = lmfit.Parameter('mu_1', value=mu1,expr='mu_0*(6716./6731.)')
    gh = lmfit.Parameter('sig_1', value=sig1,expr='sig_0')
    hi = lmfit.Parameter('amp_1', value=amp1)
    ij = lmfit.Parameter('mu_2', value=mu2,expr='mu_0*(6583./6731.)')
    jk = lmfit.Parameter('sig_2', value=sig2,expr='sig_0')
    kl = lmfit.Parameter('amp_2', value=amp2)
    lm = lmfit.Parameter('mu_3', value=mu3,expr='mu_5*(6563./6300.)')		# It should be attached to the [OI] lines when available!!
    mn = lmfit.Parameter('sig_3', value=sig3,expr='sig_5')			# It should be attached to the [OI] lines when available!!
    no = lmfit.Parameter('amp_3', value=amp3)
    op = lmfit.Parameter('mu_4', value=mu4,expr='mu_0*(6548./6731.)')
    pq = lmfit.Parameter('sig_4', value=sig4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value=amp4,expr='amp_2*(1./3.)')
    rs = lmfit.Parameter('mu_5', value=mu5)
    st = lmfit.Parameter('sig_5', value=sig5)
    tu = lmfit.Parameter('amp_5', value=amp5)
    uv = lmfit.Parameter('mu_6', value=mu6,expr='mu_5*(6363./6300.)')
    vw = lmfit.Parameter('sig_6', value=sig6,expr='sig_5')
    wy = lmfit.Parameter('amp_6', value=amp6,expr='amp_5*(1./3.)')

elif meth == 'M2':
    print('If there are no OI lines in this spectra this method will not work correctly. Please select another one!')
    cd = lmfit.Parameter('mu_0', value=mu0)
    de = lmfit.Parameter('sig_0', value=sig0)
    ef = lmfit.Parameter('amp_0', value=amp0)
    fg = lmfit.Parameter('mu_1', value=mu1,expr='mu_0*(6716./6731.)')
    gh = lmfit.Parameter('sig_1', value=sig1,expr='sig_0')
    hi = lmfit.Parameter('amp_1', value=amp1)
    ij = lmfit.Parameter('mu_2', value=mu2,expr='mu_5*(6583./6300.)')
    jk = lmfit.Parameter('sig_2', value=sig2,expr='sig_5')		
    kl = lmfit.Parameter('amp_2', value=amp2)			
    lm = lmfit.Parameter('mu_3', value=mu3,expr='mu_5*(6563./6300.)')
    mn = lmfit.Parameter('sig_3', value=sig3,expr='sig_5')
    no = lmfit.Parameter('amp_3', value=amp3)			
    op = lmfit.Parameter('mu_4', value=mu4,expr='mu_5*(6548./6300.)')
    pq = lmfit.Parameter('sig_4', value=sig4,expr='sig_5')	
    qr = lmfit.Parameter('amp_4', value=amp4,expr='amp_2*(1./3.)')
    rs = lmfit.Parameter('mu_5', value=mu5)
    st = lmfit.Parameter('sig_5', value=sig5)
    tu = lmfit.Parameter('amp_5', value=amp5)
    uv = lmfit.Parameter('mu_6', value=mu6,expr='mu_5*(6363./6300.)')
    vw = lmfit.Parameter('sig_6', value=sig6,expr='sig_5')
    wy = lmfit.Parameter('amp_6', value=amp6,expr='amp_5*(1./3.)')

params.add_many(rs,st,tu,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,uv,vw,wy)
paramsbH.add_many(rs,st,tu,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,uv,vw,wy,ab,bc,yz)

# and make the fit using lmfit
resu  = sing_mod.fit(data_cor,mu=mu0,sigm=sig0,amp=amp0,x=l)
resu1 = comp_mod.fit(data_cor,params,x=l)
broadresu = broad_mod.fit(data_cor,paramsbH,x=l)

# In order to determine if the lines need one more gaussian to be fit correctly, we apply the condition
# that the std dev of the continuum should be higher than 3 times the std dev of the residuals of the 
# fit of the line. We have to calculate the stddev of the continuum in a place where there are no 
# lines (True for all AGNs spectra in this range).
# Calculate the standard deviation of a part of the continuum without lines nor contribution of them
std0   = np.where(l>input('lim inf for determining the stddev of the continuum (angs)?: '))[0][0]
std1   = np.where(l<input('lim sup for determining the stddev of the continuum (angs)?: '))[0][-1]
stadev = np.std(data_cor[std0:std1]) 

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
std_o1 = np.std(resu1.residual[np.where(l<l11)[0][-1]:np.where(l>l12)[0][-1]])
std_o2 = np.std(resu1.residual[np.where(l<l13)[0][-1]:np.where(l>l14)[0][-1]])
# one component + broad halpha
stdb_s2 = np.std(broadresu.residual[np.where(l<l1)[0][-1]:np.where(l>l2)[0][-1]])
stdb_s1 = np.std(broadresu.residual[np.where(l<l3)[0][-1]:np.where(l>l4)[0][-1]])
stdb_n2 = np.std(broadresu.residual[np.where(l<l5)[0][-1]:np.where(l>l6)[0][-1]])
stdb_ha = np.std(broadresu.residual[np.where(l<l7)[0][-1]:np.where(l>l8)[0][-1]])
stdb_n1 = np.std(broadresu.residual[np.where(l<l9)[0][-1]:np.where(l>l10)[0][-1]])
stdb_o1 = np.std(broadresu.residual[np.where(l<l11)[0][-1]:np.where(l>l12)[0][-1]])
stdb_o2 = np.std(broadresu.residual[np.where(l<l13)[0][-1]:np.where(l>l14)[0][-1]])

##################################################################################################################
##################################################################################################################
########################################### RESULTS: PLOT and PRINT ##############################################
##################################################################################################################
##################################################################################################################
#
# Now we create the individual gaussians in order to plot and print the results
n = 0
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
print(resu1.params['mu_5'])
print(resu1.params['sig_5'])
print(resu1.params['amp_5'])
print(resu1.params['mu_6'])
print(resu1.params['sig_6'])
print(resu1.params['amp_6'])
print('')
print('The chi-square of the fit is: {:.5f}'.format(resu1.chisqr))
#print('The reduced chi-square of the fit is: {:.5f}'.format(resu1.redchi))
print('')
#print('The standard deviation of the continuum is: {:.5f}  and the one of the SII line is: {:.5f}'.format(stadev, std_line))
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> ')
print('		str(std_s2)+' < 3*'+str(stadev)')
print('		str(std_s1)+' < 3*'+str(stadev)')
print('		str(std_n2)+' < 3*'+str(stadev)')
print('		str(std_ha)+' < 3*'+str(stadev)')
print('		str(std_n1)+' < 3*'+str(stadev)')
print('		str(std_o1)+' < 3*'+str(stadev)')
print('		str(std_o2)+' < 3*'+str(stadev)')

# Now we create and plot the individual gaussians of the fit
gaus1 = gaussian(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0']) 
gaus2 = gaussian(l,resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'])
gaus3 = gaussian(l,resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2'])
gaus4 = gaussian(l,resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'])
gaus5 = gaussian(l,resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'])
gaus6 = gaussian(l,resu1.values['mu_5'],resu1.values['sig_5'],resu1.values['amp_5'])
gaus7 = gaussian(l,resu1.values['mu_6'],resu1.values['sig_6'],resu1.values['amp_6'])

# Save the data from the fit in a .txt
with open('fit_result.txt', 'w') as fh:
    fh.write(resu1.fit_report())
    print('The results have been saved in a file. Please check it!')

################################################ PLOT ######################################################
plt.close()
# MAIN plot
fig1   = plt.figure(1)
frame1 = fig1.add_axes((.1,.3,.8,.6))	 	# xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)				# Initial data
plt.plot(l,funcgauslin(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0'],
		       resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'],
		       resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2'],
		       resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'],
		       resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'],
		       resu1.values['mu_5'],resu1.values['sig_5'],resu1.values['amp_5'],
		       resu1.values['mu_6'],resu1.values['sig_6'],resu1.values['amp_6']),'r--')
plt.plot(l,gaus1,'c--')
plt.plot(l,gaus2,'c--')
plt.plot(l,gaus3,'c--')
plt.plot(l,gaus4,'c--')
plt.plot(l,gaus5,'c--',label='N')
plt.plot(l,gaus6,'c--')
plt.plot(l,gaus7,'c--')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
plt.plot(l[std0:std1],data_cor[std0:std1],'g')		# Zone where the continuum stddev is calculated
#plt.plot(l[liminf:limsup],data_cor[liminf:limsup],'g')	# Zone where the line stddev is calculated

frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend()

# Residual plot
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.plot(l,-resu1.residual,color='grey')	# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')		# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--')	# 3 sigma down limit


##########################################################################################################################
############################################# PRINT AND PLOT 1 COMP + HA #################################################
#
# Now we create the individual gaussians in order to plot and print the results
n = 0
print('			RESULTS OF THE FIT 1 COMP + BROAD Halpha: ')
print('Linear fit equation: {:.5f}*x + {:.5f}'.format(linresu.values['slope'], linresu.values['intc']))
print('')
print('The rest of the results can be displayed all together with resu1.params; the data can be accesed with resu1.values['']')
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
print(broadresu.params['mu_5'])
print(broadresu.params['sig_5'])
print(broadresu.params['amp_5'])
print(broadresu.params['mu_6'])
print(broadresu.params['sig_6'])
print(broadresu.params['amp_6'])
print('')
print('The chi-square of the fit is: {:.5f}'.format(broadresu.chisqr))
print('')
#print('The standard deviation of the continuum is: {:.5f}  and the one of the SII line is: {:.5f}'.format(stadev, std_line))
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> ')
print('		str(stdb_s2)+' < 3*'+str(stadev)')
print('		str(stdb_s1)+' < 3*'+str(stadev)')
print('		str(stdb_n2)+' < 3*'+str(stadev)')
print('		str(stdb_ha)+' < 3*'+str(stadev)')
print('		str(stdb_n1)+' < 3*'+str(stadev)')
print('		str(stdb_o1)+' < 3*'+str(stadev)')
print('		str(stdb_o2)+' < 3*'+str(stadev)')

finalb_fit = funcbroad(l,broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0'],
		       broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'],
		       broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2'],
		       broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'],
		       broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'],
		       broadresu.values['mu_5'],broadresu.values['sig_5'],broadresu.values['amp_5'],
		       broadresu.values['mu_6'],broadresu.values['sig_6'],broadresu.values['amp_6'],
		       broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])

# Now we create and plot the individual gaussians of the fit
gausb1 = gaussian(l,broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0']) 
gausb2 = gaussian(l,broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'])
gausb3 = gaussian(l,broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2'])
gausb4 = gaussian(l,broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'])
gausb5 = gaussian(l,broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'])
gausb6 = gaussian(l,broadresu.values['mu_5'],broadresu.values['sig_5'],broadresu.values['amp_5'])
gausb7 = gaussian(l,broadresu.values['mu_6'],broadresu.values['sig_6'],broadresu.values['amp_6'])
gausb8 = gaussian(l,broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])

# Save the data from the fit in a .txt
with open('fitb_result.txt', 'w') as fh:
    fh.write(broadresu.fit_report())
    print('The results have been saved in a file. Please check it!')

################################################ PLOT ######################################################
plt.figure()
# MAIN plot
fig2   = plt.figure(2)
frame3 = fig2.add_axes((.1,.3,.8,.6))	 	# xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)				# Initial data
plt.plot(l,finalb_fit,'r--')
plt.plot(l,gausb1,'c--')
plt.plot(l,gausb2,'c--')
plt.plot(l,gausb3,'c--')
plt.plot(l,gausb4,'c--')
plt.plot(l,gausb5,'c--',label='N')
plt.plot(l,gausb6,'c--')
plt.plot(l,gausb7,'c--')
plt.plot(l,gausb8,'m--',label='B')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
plt.plot(l[std0:std1],data_cor[std0:std1],'g')		# Zone where the continuum stddev is calculated

frame3.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend()

# Residual plot
frame4 = fig2.add_axes((.1,.1,.8,.2))
plt.plot(l,data_cor-finalb_fit,color='grey')	# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')		# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--')	# 3 sigma down limit
