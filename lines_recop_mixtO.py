'''
This script makes a gaussian fit to the emission lines of AGN spectra
It is needed a path, the spectrum in which the fit is going to be made and the initial estimation of the fit
This script only applies the mixture methods to make the fit
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import Ofuncts
from plot_refer_lines import refer_plot
from plot_broad_mixtO import broad_plot
from PyAstronomy.pyasl import ftest
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
l_Halpha = 6562.801
l_NII_1  = 6548.05
l_NII_2  = 6583.45
l_SII_1  = 6716.44
l_SII_2  = 6730.82
l_OI_1 = 6300.304
l_OI_2 = 6363.77

# Constants and STIS parameters
v_luz = 299792.458 # km/s
plate_scale = data_head['PLATESC']
fwhm = 2*np.sqrt(2*np.log(2)) # por sigma
if plate_scale == 0.05078:
    siginst = 1.1	# A if binning 1x1 // 2.2 if binning 1x2
    sig_inst = siginst/fwhm	# considering a gaussian, same units as in the fit
    ang_to_pix = 0.554
    pix_to_v = 25	# km/s
    #minbroad = 24.
elif plate_scale == 0.10156:
    siginst = 2.2	# A if binning 1x1 // 2.2 if binning 1x2
    sig_inst = siginst/fwhm	# considering a gaussian, same units as in the fit
    ang_to_pix = 1.108
    pix_to_v = 47	# km/s
    #minbroad = 12.83402
sig_inicial = sig_inst + 0.5
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
    l11 = input('lambda inf for OI 1 (angs)?: ')
    l12 = input('lambda sup for OI 1 (angs)?: ')
    l13 = input('lambda inf for OI 2 (angs)?: ')
    l14 = input('lambda sup for OI 2 (angs)?: ')
    z = input('Redshift of the galaxy?: ')
    erz = input('Error of the redshift of the galaxy?: ')
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
    l11 = t[10,]
    l12 = t[11,]
    l13 = t[12,]
    l14 = t[13,]
    z = t[14,]
    erz = t[15,]

# Systemic velocity of the galaxy
vsys = v_luz*z
er_vsys = v_luz*erz

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
newx6 = l[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]+1]	# OI1
newy6 = data_cor[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]+1]
newx7 = l[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]+1]	# OI2
newy7 = data_cor[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]+1]

# Initial guesses of the fitting parameters
sig0 = sig_inicial			# SII2
sig20 = 3.
mu0  = newx1[np.argmax(newy1)]
amp0 = max(newy1)
amp20 = max(newy1)/2.
sig1 = sig_inicial			# SII1
sig21 = 3.
mu1 = newx2[np.argmax(newy2)]
amp1 = max(newy2)
amp21 = max(newy2)/2.
sig2 = sig_inicial			# NII2
sig22 = 10.
mu2 = newx3[np.argmax(newy3)]
amp2 = max(newy3)
amp22 = max(newy3)/2.
sig3 = sig_inicial			# Halpha
sig23 = 10.
mu3 = newx4[np.argmax(newy4)]
amp3 = max(newy4)
amp23 = max(newy4)/2.
sig4 = sig_inicial			# NII1
sig24 = 10.
mu4 = newx5[np.argmax(newy5)]
amp4 = max(newy5)
amp24 = max(newy5)/2.
sig5 = sig_inicial			# OI1
sig25 = 3.
mu5  = newx6[np.argmax(newy6)]
amp5 = max(newy6)
amp25 = max(newy6)/2.
sig6 = sig_inicial			# OI2
sig26 = 3.
mu6  = newx7[np.argmax(newy7)]
amp6 = max(newy7)
amp26 = max(newy7)/2.

# Start the parameters for the LINEAR fit
in_slope = 0.
in_intc  = data_cor[0]

# Redefine the lambda zone with the first and last point and the zones in between OI2-NII1 and NII2-SII1
newl = l[1:3]
zone_O_N = l[np.where(l<6400.)[0][-1]+10:np.where(l>l9)[0][0]-10]
zone_N_S = l[np.where(l<l6)[0][-1]+30:np.where(l>l3)[0][0]-30]
newl = np.append(newl,zone_O_N)
newl = np.append(newl,zone_N_S)
newl = np.append(newl,l[-15:-1])
# now we do the same but with the flux data (y vector)
newflux = data_cor[1:3]
zon_O_N = data_cor[np.where(l<6400.)[0][-1]+10:np.where(l>l9)[0][0]-10]
zon_N_S = data_cor[np.where(l<l6)[0][-1]+30:np.where(l>l3)[0][0]-30]
newflux = np.append(newflux,zon_O_N)
newflux = np.append(newflux,zon_N_S)
newflux = np.append(newflux,data_cor[-15:-1])

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
lin_mod = lmfit.Model(Ofuncts.linear)
one_mod = lmfit.Model(Ofuncts.twogaussian)
two_mod = lmfit.Model(Ofuncts.funcSII2comp)

# and initialise the model in the whole spectra for several different models
comp_mod = lmfit.Model(Ofuncts.funcgauslin)
broad_mod = lmfit.Model(Ofuncts.funcbroad)
twocomp_mod = lmfit.Model(Ofuncts.func2com)
twobroadcomp_mod = lmfit.Model(Ofuncts.func2bcom)

# We make the linear fit only with some windows of the spectra, and calculate the line to introduce it in the formula
linresu  = lin_mod.fit(newflux,slope=in_slope,intc=in_intc,x=newl)
new_slop = linresu.values['slope']
new_intc = linresu.values['intc']

# Now we define the initial guesses and the constraints
params1s = lmfit.Parameters()
params1o = lmfit.Parameters()
params2s = lmfit.Parameters()
params2o = lmfit.Parameters()

sl = lmfit.Parameter('slop', value=new_slop,vary=False)
it = lmfit.Parameter('intc', value=new_intc,vary=False)
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
meth = input('Which mixt method to be applied? ("M1"/"M2"): ')	# Method to fit

print('The fit will be doing by taking SII and OI lines as the references')
cds = lmfit.Parameter('mu_0', value=mu0)
des = lmfit.Parameter('sig_0', value=sig0,min=sig_inst)
efs = lmfit.Parameter('amp_0', value=amp0,min=0.)
fgs = lmfit.Parameter('mu_1', value=mu1,expr='mu_0*(6716.44/6730.82)')
ghs = lmfit.Parameter('sig_1', value=sig1,expr='sig_0')
his = lmfit.Parameter('amp_1', value=amp1,min=0.)
# second components
aaas = lmfit.Parameter('mu_20', value=mu0)
aabs = lmfit.Parameter('sig_20', value=sig20,min=sig_inst)
aacs = lmfit.Parameter('amp_20', value=amp20,min=0.)
aads = lmfit.Parameter('mu_21', value=mu1,expr='mu_20*(6716.44/6730.82)')
aaes = lmfit.Parameter('sig_21', value=sig21,min=sig_inst,expr='sig_20')
aafs = lmfit.Parameter('amp_21', value=amp21,min=0.)

cdo = lmfit.Parameter('mu_0', value=mu5)
deo = lmfit.Parameter('sig_0', value=sig5,min=sig_inst)
efo = lmfit.Parameter('amp_0', value=amp5,min=0.)
fgo = lmfit.Parameter('mu_1', value=mu6,expr='mu_0*(6363.77/6300.304)')
gho = lmfit.Parameter('sig_1', value=sig6,expr='sig_0')
hio = lmfit.Parameter('amp_1', value=amp6,min=0.,expr='amp_0*(1./3.)')
# second components
aaao = lmfit.Parameter('mu_20', value=mu5)
aabo = lmfit.Parameter('sig_20', value=sig25,min=sig_inst)
aaco = lmfit.Parameter('amp_20', value=amp25,min=0.)
aado = lmfit.Parameter('mu_21', value=mu6,expr='mu_0*(6363.77/6300.304)')
aaeo = lmfit.Parameter('sig_21', value=sig26,min=sig_inst,expr='sig_20')
aafo = lmfit.Parameter('amp_21', value=amp26,min=0.,expr='amp_20*(1./3.)')
# Zone of the lines for doing the fit
xs_one = l[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]
ys_one = data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]
xo_one = l[np.where(l>l11)[0][0]:np.where(l<l14)[0][-1]+20]
yo_one = data_cor[np.where(l>l11)[0][0]:np.where(l<l14)[0][-1]+20]
# add a sequence of Parameters
params1s.add_many(sl,it,cds,des,efs,fgs,ghs,his)
params1o.add_many(sl,it,cdo,deo,efo,fgo,gho,hio)
params2s.add_many(sl,it,cds,des,efs,fgs,ghs,his,aaas,aabs,aacs,aads,aaes,aafs)
params2o.add_many(sl,it,cdo,deo,efo,fgo,gho,hio,aaao,aabo,aaco,aado,aaeo,aafo)

###################################################################################################################
# Make the fit using lmfit
oneSresu = one_mod.fit(ys_one,params1s,x=xs_one)
oneOresu = one_mod.fit(yo_one,params1o,x=xo_one)
twoSresu = two_mod.fit(ys_one,params2s,x=xs_one)
twoOresu = two_mod.fit(yo_one,params2o,x=xo_one)

lmfit.model.save_modelresult(oneSresu, path+'Mix_SII_onecomp_modelresult.sav')
lmfit.model.save_modelresult(oneOresu, path+'Mix_OI_onecomp_modelresult.sav')
lmfit.model.save_modelresult(twoSresu, path+'Mix_SII_twocomps_modelresult.sav')
lmfit.model.save_modelresult(twoOresu, path+'Mix_OI_twocomps_modelresult.sav')
with open(path+'fit_mix_oneSII_result.txt', 'w') as fh:
    fh.write(oneSresu.fit_report())
with open(path+'fit_mix_oneOI_result.txt', 'w') as fh:
    fh.write(oneOresu.fit_report())
with open(path+'fit_mix_twoSII_result.txt', 'w') as fh:
    fh.write(twoSresu.fit_report())
with open(path+'fit_mix_twoOI_result.txt', 'w') as fh:
    fh.write(twoOresu.fit_report())

##################################### PLOT and PRINT for the SII lines #################################################
#
# Now we create the individual gaussians in order to plot and print the results for only 1 component
print('				RESULTS OF THE FIT: ')
print('Linear fit equation: {:.5f}*x + {:.5f}'.format(linresu.values['slope'], linresu.values['intc']))
print('')
print('The rest of the results can be displayed all together with two/SIIresu.params; the data can be accesed with two/SIIresu.values['']')
print('')
print('The chi-square of the fit for 1 gaussian for SII is: {:.5f}'.format(oneSresu.chisqr))
print('The chi-square of the fit for 1 gaussian for OI is: {:.5f}'.format(oneOresu.chisqr))
print('The chi-square of the fit for 2 gaussian for SII is: {:.5f}'.format(twoSresu.chisqr))
print('The chi-square of the fit for 2 gaussian for OI is: {:.5f}'.format(twoOresu.chisqr))
print('')

# Now we create and plot the individual gaussians of the fit
gausS1 = Ofuncts.gaussian(l,oneSresu.values['mu_0'],oneSresu.values['sig_0'],oneSresu.values['amp_0']) 
gausS2 = Ofuncts.gaussian(l,oneSresu.values['mu_1'],oneSresu.values['sig_1'],oneSresu.values['amp_1'])
gausO1 = Ofuncts.gaussian(l,oneOresu.values['mu_0'],oneOresu.values['sig_0'],oneOresu.values['amp_0']) 
gausO2 = Ofuncts.gaussian(l,oneOresu.values['mu_1'],oneOresu.values['sig_1'],oneOresu.values['amp_1'])

gausS21 = Ofuncts.gaussian(l,twoSresu.values['mu_0'],twoSresu.values['sig_0'],twoSresu.values['amp_0']) 
gausS22 = Ofuncts.gaussian(l,twoSresu.values['mu_1'],twoSresu.values['sig_1'],twoSresu.values['amp_1'])
gausS23 = Ofuncts.gaussian(l,twoSresu.values['mu_20'],twoSresu.values['sig_20'],twoSresu.values['amp_20'])
gausS24 = Ofuncts.gaussian(l,twoSresu.values['mu_21'],twoSresu.values['sig_21'],twoSresu.values['amp_21'])
gausO21 = Ofuncts.gaussian(l,twoOresu.values['mu_0'],twoOresu.values['sig_0'],twoOresu.values['amp_0']) 
gausO22 = Ofuncts.gaussian(l,twoOresu.values['mu_1'],twoOresu.values['sig_1'],twoOresu.values['amp_1'])
gausO23 = Ofuncts.gaussian(l,twoOresu.values['mu_20'],twoOresu.values['sig_20'],twoOresu.values['amp_20'])
gausO24 = Ofuncts.gaussian(l,twoOresu.values['mu_21'],twoOresu.values['sig_21'],twoOresu.values['amp_21'])

		 	  
SIIOIfin_fit = Ofuncts.funcSII2comp(l,new_slop,new_intc,
				     oneSresu.values['mu_0'],oneSresu.values['sig_0'],oneSresu.values['amp_0'],
				     oneSresu.values['mu_1'],oneSresu.values['sig_1'],oneSresu.values['amp_1'],
				     oneOresu.values['mu_0'],oneOresu.values['sig_0'],oneOresu.values['amp_0'],
				     oneOresu.values['mu_1'],oneOresu.values['sig_1'],oneOresu.values['amp_1'])

SIIOI2fin_fit = Ofuncts.funcbroad(l,new_slop,new_intc,
				     twoSresu.values['mu_0'],twoSresu.values['sig_0'],twoSresu.values['amp_0'],
				     twoSresu.values['mu_1'],twoSresu.values['sig_1'],twoSresu.values['amp_1'],
				     twoSresu.values['mu_20'],twoSresu.values['sig_20'],twoSresu.values['amp_20'],
				     twoSresu.values['mu_21'],twoSresu.values['sig_21'],twoSresu.values['amp_21'],
				     twoOresu.values['mu_0'],twoOresu.values['sig_0'],twoOresu.values['amp_0'],
				     twoOresu.values['mu_1'],twoOresu.values['sig_1'],twoOresu.values['amp_1'],
				     twoOresu.values['mu_20'],twoOresu.values['sig_20'],twoOresu.values['amp_20'],
				     twoOresu.values['mu_21'],twoOresu.values['sig_21'],twoOresu.values['amp_21'])

# one component SII
std_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]+10]-SIIOIfin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]+10])
std_s1 = np.std(data_cor[np.where(l<l3)[0][-1]-10:np.where(l>l4)[0][0]]-SIIOIfin_fit[np.where(l<l3)[0][-1]-10:np.where(l>l4)[0][0]])
std_o2 = np.std(data_cor[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]+10]-SIIOIfin_fit[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]+10])
std_o1 = np.std(data_cor[np.where(l<l11)[0][-1]-10:np.where(l>l12)[0][0]]-SIIOIfin_fit[np.where(l<l11)[0][-1]-10:np.where(l>l12)[0][0]])
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
print('		For SII2: '+str(std_s2/stadev)+' < 3')
print('		For SII1: '+str(std_s1/stadev)+' < 3')
print('		For OI2: '+str(std_o2/stadev)+' < 3')
print('		For OI1: '+str(std_o1/stadev)+' < 3')
# two components
std2_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]+10]-SIIOI2fin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]+10])
std2_s1 = np.std(data_cor[np.where(l<l3)[0][-1]-10:np.where(l>l4)[0][0]]-SIIOI2fin_fit[np.where(l<l3)[0][-1]-10:np.where(l>l4)[0][0]])
std2_o2 = np.std(data_cor[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]+10]-SIIOI2fin_fit[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]+10])
std2_o1 = np.std(data_cor[np.where(l<l11)[0][-1]-10:np.where(l>l12)[0][0]]-SIIOI2fin_fit[np.where(l<l11)[0][-1]-10:np.where(l>l12)[0][0]])
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
print('		For SII2: '+str(std2_s2/stadev)+' < 3')
print('		For SII1: '+str(std2_s1/stadev)+' < 3')
print('		For OI2: '+str(std2_o2/stadev)+' < 3')
print('		For OI1: '+str(std2_o1/stadev)+' < 3')


# We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
maxS1 = max(SIIOIfin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]]) #SIIfin_fit[np.where(abs(SIIresu.values['mu_0']-l)<0.28)[0][0]]
maxS2 = max(SIIOIfin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]]) #SIIfin_fit[np.where(abs(SIIresu.values['mu_1']-l)<0.28)[0][0]]
maxO1 = max(SIIOIfin_fit[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]])
maxO2 = max(SIIOIfin_fit[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]])
max2S1 = max(SIIOI2fin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]]) #SII2fin_fit[np.where(abs(twoSIIresu.values['mu_0']-l)<0.28)[0][0]] 
max2S2 = max(SIIOI2fin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]]) #SII2fin_fit[np.where(abs(twoSIIresu.values['mu_1']-l)<0.28)[0][0]] 
max2O1 = max(SIIOI2fin_fit[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]])
max2O2 = max(SIIOI2fin_fit[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]])
# Systemic velocity + error
vsys = v_luz*z
er_vsys = v_luz*erz
# one component
vS2 = (v_luz*((oneSresu.values['mu_0']-l_SII_2)/l_SII_2))-vsys
vO1 = (v_luz*((oneOresu.values['mu_0']-l_OI_1)/l_OI_1))-vsys
sigS2 = pix_to_v*np.sqrt(oneSresu.values['sig_0']**2-sig_inst**2)
sigO1 = pix_to_v*np.sqrt(oneOresu.values['sig_0']**2-sig_inst**2)
# two comps
v2S2 = (v_luz*((twoSresu.values['mu_0']-l_SII_2)/l_SII_2))-vsys
v2O1 = (v_luz*((twoOresu.values['mu_0']-l_OI_1)/l_OI_1))-vsys
v20S2 = (v_luz*((twoSresu.values['mu_20']-l_SII_2)/l_SII_2))-vsys
v20O1 = (v_luz*((twoOresu.values['mu_20']-l_OI_1)/l_OI_1))-vsys
sig2S2 = pix_to_v*np.sqrt(twoSresu.values['sig_0']**2-sig_inst**2)
sig2O1 = pix_to_v*np.sqrt(twoSresu.values['sig_0']**2-sig_inst**2)
sig20S2 = pix_to_v*np.sqrt(twoOresu.values['sig_20']**2-sig_inst**2)
sig20O1 = pix_to_v*np.sqrt(twoOresu.values['sig_20']**2-sig_inst**2)

if oneSresu.params['mu_0'].stderr == None: 
     print('Problem determining the errors!')
     evS2,esigS2 = 0.,0.
elif oneSresu.params['mu_0'].stderr != None: 
     evS2 = ((v_luz/l_SII_2)*oneSresu.params['mu_0'].stderr)-er_vsys
     esigS2 = pix_to_v*np.sqrt(oneSresu.values['sig_0']*oneSresu.params['sig_0'].stderr)/(np.sqrt(oneSresu.values['sig_0']**2-sig_inst**2))

if oneOresu.params['mu_0'].stderr == None: 
     print('Problem determining the errors!')
     evO1,esigO1 = 0.,0.
elif oneOresu.params['mu_0'].stderr != None: 
     evO1 = ((v_luz/l_OI_1)*oneOresu.params['mu_0'].stderr)-er_vsys
     esigO1 = pix_to_v*np.sqrt(oneOresu.values['sig_0']*oneOresu.params['sig_0'].stderr)/(np.sqrt(oneOresu.values['sig_0']**2-sig_inst**2))

if twoSresu.params['mu_20'].stderr == None:
    print('Problem determining the errors!')
    ev20S2,ev2S2,esig2S2,esig20S2 = 0.,0.,0.,0.
elif twoSresu.params['mu_20'].stderr != None:
    ev2S2 = ((v_luz/l_SII_2)*twoSresu.params['mu_0'].stderr)-er_vsys
    ev20S2 = ((v_luz/l_SII_2)*twoSresu.params['mu_20'].stderr)-er_vsys
    esig2S2 = pix_to_v*np.sqrt(twoSresu.values['sig_0']*twoSresu.params['sig_0'].stderr)/(np.sqrt(twoSresu.values['sig_0']**2-sig_inst**2))
    esig20S2 = pix_to_v*np.sqrt(twoSresu.values['sig_20']*twoSresu.params['sig_20'].stderr)/(np.sqrt(twoSresu.values['sig_20']**2-sig_inst**2))

if twoOresu.params['mu_20'].stderr == None:
    print('Problem determining the errors!')
    ev20O1,ev2O1,esig2O1,esig20O1 = 0.,0.,0.,0.
elif twoOresu.params['mu_20'].stderr != None:
    ev2O1 = ((v_luz/l_OI_1)*twoOresu.params['mu_0'].stderr)-er_vsys
    ev20O1 = ((v_luz/l_OI_1)*twoOresu.params['mu_20'].stderr)-er_vsys
    esig2O1 = pix_to_v*np.sqrt(twoOresu.values['sig_0']*twoOresu.params['sig_0'].stderr)/(np.sqrt(twoOresu.values['sig_0']**2-sig_inst**2))
    esig20O1 = pix_to_v*np.sqrt(twoOresu.values['sig_20']*twoOresu.params['sig_20'].stderr)/(np.sqrt(twoOresu.values['sig_20']**2-sig_inst**2))

################################################### PLOT #########################################################
plt.close()
# MAIN plot
fig1   = plt.figure(1,figsize=(10, 9))
frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,SIIOIfin_fit,'r-')
plt.plot(l,gausS1,'c--')
plt.plot(l,gausO1,'c--')
plt.plot(l,gausS2,'c--',label='N')
plt.plot(l,gausO2,'c--')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
		    r'$V_{OI_{1}}$ = '+ '{:.2f} +- {:.2f}'.format(vO1,evO1),
		    r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    r'$\sigma_{OI_{1}}$ = '+ '{:.2f} +- {:.2f}'.format(sigO1,esigO1),
		    r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(maxS2/maxS1),
		    r'$\frac{F_{OI_{2}}}{F_{OI_{1}}}$ = '+ '{:.3f}'.format(maxO2/maxO1)))
#		    r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxS1)+' $10^{-14}$'))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
if l11<6900.:
    x_frame = 6380.
else: 
    x_frame = 6800.
frame1.text(x_frame,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)

frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.plot(l,data_cor-SIIOIfin_fit,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
plt.ylim(-(3*stadev)*2,(3*stadev)*2)
plt.savefig(path+'adj_metmix_SIIOI_1comp.png')

###############################################################################################################
# Two components in SII
# MAIN plot
fig2   = plt.figure(2,figsize=(10, 9))
frame3 = fig2.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,SIIOI2fin_fit,'r-')
plt.plot(l,gausS21,'c--')
plt.plot(l,gausS22,'c--',label='N')
plt.plot(l,gausO21,'c--')
plt.plot(l,gausO22,'c--')
plt.plot(l,gausS23,'m--')
plt.plot(l,gausS24,'m--',label='S')
plt.plot(l,gausO23,'m--')
plt.plot(l,gausO24,'m--')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
textstr = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2S2,ev2S2),
		    r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v20S2,ev20S2),
		    r'$V_{OI_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2O1,ev2O1),
		    r'$V_{OI_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v20O1,ev20O1),
		    r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2S2,esig2S2),
		    r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig20S2,esig20S2),
		    r'$\sigma_{OI_{1-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2O1,esig2O1),
		    r'$\sigma_{OI_{1-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig20O1,esig20O1),
		    r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(max2S2/max2S1),
		    r'$\frac{F_{OI_{2}}}{F_{OI_{1}}}$ = '+ '{:.3f}'.format(max2O2/max2O1)))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
frame3.text(x_frame,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

frame3.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame4 = fig2.add_axes((.1,.1,.8,.2))
plt.plot(l,data_cor-SIIOI2fin_fit,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
plt.ylim(-(3*stadev)*2,(3*stadev)*2)
plt.savefig(path+'adj_metmixt_SIIOI_2comp.png')

##########################################################################################################################################################
# We make an F-test to see if it is significant the presence of a second component in the lines. 
# As the only possible method here is the S-method due to the fact that there are no O-lines in this spectra, 
# then the method can only be applied to the SII lines (so the wavelength range would be around this two lines)
pre_Sx = data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-SIIOIfin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]
pre_Sy = data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-SIIOI2fin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]
pre_Ox = data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-SIIOIfin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]
pre_Oy = data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-SIIOI2fin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]
tSx, tSy = stats.obrientransform(pre_Sx, pre_Sy)
fvalueS1, pvalueS1 = stats.f_oneway(tSx,tSy)
fvalueS, pvalueS = stats.f_oneway(pre_Sx,pre_Sy)
fvalueS2, pvalueS2 = stats.levene(pre_Sx,pre_Sy)
##########################################################################################################################################################
print('')
if oneSresu.chisqr < twoSresu.chisqr: 
    print('The probability of a second component (one component vs two components) in this spectra is NOT CALCULATED. ONLY ONE COMPONENT AVAILABLE')
    print('Chi-square of the two component-fit bigger than the chi-square of the one component fit')
else: 
    fstat = ftest(oneSresu.chisqr,twoSresu.chisqr,oneSresu.nfree,twoSresu.nfree)
    print('')
    print('The probability of a second component in SII2 (one component vs two components) in this spectra with the F-test of IDL is: '+str(fstat['p-value']))
    print('The probability of a second component in SII2 (one component vs two components) in this spectra with the F-test is: '+str(pvalueS))
    print('The probability of a second component in SII2 (one component vs two components) in this spectra with the F-test (and O Brien) is: '+str(pvalueS1))
    print('The probability of a second component in SII2 (one component vs two components) in this spectra with the Levene-test is: '+str(pvalueS2))
    print('')

##########################################################################################################################################################
#---------------------------------------------------------------------------------------------------------------------------------------------------------
##########################################################################################################################################################

# Select if one or two components in the reference lines and then apply to the rest
if (oneSresu.chisqr < twoSresu.chisqr) and (oneOresu.chisqr < twoOresu.chisqr): 
    trigger = 'Y'
else: 
    trigger = input('Is the fit good enough with one component? ("Y"/"N"): ')

if trigger == 'Y':
    params = lmfit.Parameters()
    if meth == 'M1':
	# Now we define the initial guesses and the constraints
	cd = lmfit.Parameter('mu_0', value = oneSresu.values["mu_0"],vary=False)
	de = lmfit.Parameter('sig_0', value = oneSresu.values["sig_0"],vary=False)
	ef = lmfit.Parameter('amp_0', value = oneSresu.values["amp_0"],vary=False)
	fg = lmfit.Parameter('mu_1', value = oneSresu.values["mu_1"],vary=False)
	gh = lmfit.Parameter('sig_1', value = oneSresu.values["sig_1"],vary=False)
	hi = lmfit.Parameter('amp_1', value = oneSresu.values["amp_1"],vary=False)
	rs = lmfit.Parameter('mu_5', value = oneOresu.values["mu_0"],vary=False)
	st = lmfit.Parameter('sig_5', value = oneOresu.values["sig_0"],vary=False)
	tu = lmfit.Parameter('amp_5', value = oneOresu.values["amp_0"],vary=False)
	uv = lmfit.Parameter('mu_6', value = oneOresu.values["mu_1"],vary=False)
	vw = lmfit.Parameter('sig_6', value = oneOresu.values["sig_1"],vary=False)
	wy = lmfit.Parameter('amp_6', value = oneOresu.values["amp_1"],vary=False)
	ij = lmfit.Parameter('mu_2', value = mu2,expr='mu_0*(6583.45/6730.82)')
	jk = lmfit.Parameter('sig_2', value = sig2,expr='sig_0')
	kl = lmfit.Parameter('amp_2', value = amp2,min=0.)
	lm = lmfit.Parameter('mu_3', value = mu3,expr='mu_5*(6562.801/6300.304)')
	mn = lmfit.Parameter('sig_3', value = sig3,expr='sig_5')
	no = lmfit.Parameter('amp_3', value = amp3,min=0.)
	op = lmfit.Parameter('mu_4', value = mu4,expr='mu_0*(6548.05/6730.82)')
	pq = lmfit.Parameter('sig_4', value = sig4,expr='sig_0')
	qr = lmfit.Parameter('amp_4', value = amp4,min=0.,expr='amp_2*(1./3.)',vary=False)

    elif meth == 'M2':
	# Now we define the initial guesses and the constraints
	cd = lmfit.Parameter('mu_0', value = oneSresu.values["mu_0"],vary=False)
	de = lmfit.Parameter('sig_0', value = oneSresu.values["sig_0"],vary=False)
	ef = lmfit.Parameter('amp_0', value = oneSresu.values["amp_0"],vary=False)
	fg = lmfit.Parameter('mu_1', value = oneSresu.values["mu_1"],vary=False)
	gh = lmfit.Parameter('sig_1', value = oneSresu.values["sig_1"],vary=False)
	hi = lmfit.Parameter('amp_1', value = oneSresu.values["amp_1"],vary=False)
	rs = lmfit.Parameter('mu_5', value = oneOresu.values["mu_0"],vary=False)
	st = lmfit.Parameter('sig_5', value = oneOresu.values["sig_0"],vary=False)
	tu = lmfit.Parameter('amp_5', value = oneOresu.values["amp_0"],vary=False)
	uv = lmfit.Parameter('mu_6', value = oneOresu.values["mu_1"],vary=False)
	vw = lmfit.Parameter('sig_6', value = oneOresu.values["sig_1"],vary=False)
	wy = lmfit.Parameter('amp_6', value = oneOresu.values["amp_1"],vary=False)
	ij = lmfit.Parameter('mu_2', value = mu2,expr='mu_5*(6583.45/6300.304)')
	jk = lmfit.Parameter('sig_2', value = sig2,expr='sig_5')
	kl = lmfit.Parameter('amp_2', value = amp2,min=0.)
	lm = lmfit.Parameter('mu_3', value = mu3,expr='mu_5*(6562.801/6300.304)')
	mn = lmfit.Parameter('sig_3', value = sig3,expr='sig_5')
	no = lmfit.Parameter('amp_3', value = amp3,min=0.)
	op = lmfit.Parameter('mu_4', value = mu4,expr='mu_5*(6548.05/6300.304)')
	pq = lmfit.Parameter('sig_4', value = sig4,expr='sig_5')
	qr = lmfit.Parameter('amp_4', value = amp4,min=0.,expr='amp_2*(1./3.)',vary=False)


    params.add_many(sl,it,cd,de,ef,fg,gh,hi,rs,st,tu,uv,vw,wy,ij,jk,kl,lm,mn,no,op,pq,qr)
    resu1 = comp_mod.fit(data_cor,params,x=l)
    lmfit.model.save_modelresult(resu1, path+'one_meth'+str(meth)+'_modelresult.sav')
    with open(path+'fit_meth'+str(meth)+'onecomp_result.txt', 'w') as fh:
         fh.write(resu1.fit_report())
    
    ######################################## Calculate gaussians and final fit ################################################
    # Now we create and plot the individual gaussians of the fit
    gaus1 = Ofuncts.gaussian(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0']) 
    gaus2 = Ofuncts.gaussian(l,resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'])
    gaus3 = Ofuncts.gaussian(l,resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2']) 
    gaus4 = Ofuncts.gaussian(l,resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'])
    gaus5 = Ofuncts.gaussian(l,resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'])
    gaus6 = Ofuncts.gaussian(l,resu1.values['mu_5'],resu1.values['sig_5'],resu1.values['amp_5'])
    gaus7 = Ofuncts.gaussian(l,resu1.values['mu_6'],resu1.values['sig_6'],resu1.values['amp_6'])
    fin_fit = Ofuncts.funcgauslin(l,new_slop,new_intc,
				  resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0'],
				  resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'],
				  resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2'],
				  resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'],
				  resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'],
				  resu1.values['mu_5'],resu1.values['sig_5'],resu1.values['amp_5'],
				  resu1.values['mu_6'],resu1.values['sig_6'],resu1.values['amp_6'])
      
    # one component
    stdf_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]+10]-fin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]+10])
    stdf_s1 = np.std(data_cor[np.where(l<l3)[0][-1]-10:np.where(l>l4)[0][0]]-fin_fit[np.where(l<l3)[0][-1]-10:np.where(l>l4)[0][0]])
    stdf_n2 = np.std(data_cor[np.where(l<l5)[0][-1]:np.where(l>l6)[0][0]+10]-fin_fit[np.where(l<l5)[0][-1]:np.where(l>l6)[0][0]+10])
    stdf_ha = np.std(data_cor[np.where(l<l7)[0][-1]:np.where(l>l8)[0][0]]-fin_fit[np.where(l<l7)[0][-1]:np.where(l>l8)[0][0]])
    stdf_n1 = np.std(data_cor[np.where(l<l9)[0][-1]-10:np.where(l>l10)[0][0]]-fin_fit[np.where(l<l9)[0][-1]-10:np.where(l>l10)[0][0]])
    stdf_o1 = np.std(data_cor[np.where(l<l11)[0][-1]-5:np.where(l>l12)[0][0]+10]-fin_fit[np.where(l<l11)[0][-1]-5:np.where(l>l12)[0][0]+10])
    stdf_o2 = np.std(data_cor[np.where(l<l13)[0][-1]-10:np.where(l>l14)[0][0]+10]-fin_fit[np.where(l<l13)[0][-1]-10:np.where(l>l14)[0][0]+10])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 components is... ')
    print('		For SII2: '+str(stdf_s2/stadev)+' < 3')
    print('		For SII1: '+str(stdf_s1/stadev)+' < 3')
    print('		For NII2: '+str(stdf_n2/stadev)+' < 3')
    print('		For Halpha: '+str(stdf_ha/stadev)+' < 3')
    print('		For NII1: '+str(stdf_n1/stadev)+' < 3')
    print('		For OI2: '+str(stdf_o2/stadev)+' < 3')
    print('		For OI2: '+str(stdf_o1/stadev)+' < 3')
	
    if os.path.exists(path+'eps_adj'+str(meth)+'_1.txt'): os.remove(path+'eps_adj'+str(meth)+'_1.txt')
    np.savetxt(path+'eps_adj'+str(meth)+'_1.txt',np.c_[stdf_s2/stadev,stdf_s1/stadev,stdf_n2/stadev,stdf_ha/stadev,stdf_n1/stadev,stdf_o1/stadev,stdf_o2/stadev,resu1.chisqr],
    		('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('SII2\tSII1\tNII2\tHa\tNII1\tOI1\tOI2\tChi2'))

    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    maxfS1 = fin_fit[np.where(abs(resu1.values['mu_0']-l)<0.28)[0][0]] #max(fin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    maxfS2 = fin_fit[np.where(abs(resu1.values['mu_1']-l)<0.28)[0][0]] #max(fin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    maxfN1 = fin_fit[np.where(abs(resu1.values['mu_2']-l)<0.28)[0][0]] #max(fin_fit[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
    maxfHa = fin_fit[np.where(abs(resu1.values['mu_3']-l)<0.28)[0][0]] #max(fin_fit[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
    maxfN2 = fin_fit[np.where(abs(resu1.values['mu_4']-l)<0.28)[0][0]] #max(fin_fit[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])
    maxfO1 = fin_fit[np.where(abs(resu1.values['mu_5']-l)<0.28)[0][0]] #max(fin_fit[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]])
    maxfO2 = fin_fit[np.where(abs(resu1.values['mu_6']-l)<0.28)[0][0]] #max(fin_fit[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]])

    # one component
    vS2 = (v_luz*((resu1.values['mu_0']-l_SII_2)/l_SII_2))-vsys
    vO1 = (v_luz*((resu1.values['mu_5']-l_OI_1)/l_OI_1))-vsys
    sigS2 = pix_to_v*np.sqrt(resu1.values['sig_0']**2-sig_inst**2)
    sigO1 = pix_to_v*np.sqrt(resu1.values['sig_5']**2-sig_inst**2)

    if oneSresu.params['sig_0'].stderr == None: 
	print('Problem determining the errors! First component sigma SII')
	esigS2 = 0.
    elif oneSresu.params['sig_0'].stderr != None: 
	esigS2 = pix_to_v*np.sqrt(oneSresu.values['sig_0']*oneSresu.params['sig_0'].stderr)/(np.sqrt(oneSresu.values['sig_0']**2-sig_inst**2))
   
    if oneOresu.params['sig_0'].stderr == None: 
	print('Problem determining the errors! First component sigma OI')
	esigO1 = 0.
    elif oneOresu.params['sig_0'].stderr != None: 
	esigO1 = pix_to_v*np.sqrt(oneOresu.values['sig_0']*oneOresu.params['sig_0'].stderr)/(np.sqrt(oneOresu.values['sig_0']**2-sig_inst**2))
   
    if oneSresu.params['mu_0'].stderr == None: 
	print('Problem determining the errors! First component ')
	ev2S2 = 0.
    elif oneSresu.params['mu_0'].stderr != None: 
	evS2 = ((v_luz/l_SII_2)*oneSresu.params['mu_0'].stderr)-er_vsys

    if oneOresu.params['mu_0'].stderr == None: 
	print('Problem determining the errors! First component ')
	ev2O1 = 0.
    elif oneOresu.params['mu_0'].stderr != None: 
	evO1 = ((v_luz/l_OI_1)*oneOresu.params['mu_0'].stderr)-er_vsys
    
    ################################################ PLOT ######################################################
    plt.close('all')
    # MAIN plot
    fig1   = plt.figure(1,figsize=(10, 9))
    frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(l,data_cor)			     # Initial data
    plt.plot(l,fin_fit,'r-')
    plt.plot(l,gaus1,'c--')
    plt.plot(l,gaus2,'c--')
    plt.plot(l,gaus3,'c--')
    plt.plot(l,gaus4,'c--')
    plt.plot(l,gaus5,'c--')
    plt.plot(l,gaus6,'c--')
    plt.plot(l,gaus7,'c--',label='N')
    plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
    textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
    			 r'$V_{OI_{1}}$ = '+ '{:.2f} +- {:.2f}'.format(vO1,evO1),
			 r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
			 r'$\sigma_{OI_{1}}$ = '+ '{:.2f} +- {:.2f}'.format(sigO1,esigO1),
			 r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(maxfS2/maxfS1),
			 r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxfHa)+' $10^{-14}$',
			 r'$\frac{F_{OI_{2}}}{F_{OI_{1}}}$ = '+ '{:.3f}'.format(maxfO2/maxfO1)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    frame1.text(6350.,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
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
    plt.ylim(-(3*stadev)*2,(3*stadev)*2)

    plt.savefig(path+'adj_met'+str(meth)+'_full_1comp.png')
    
    ########################################################################################################################33    
    trigger2 = input('Do the fit needs a broad Halpha component? ("Y"/"N"): ')
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
	ab = lmfit.Parameter('mu_b',value=mub)#6605.,vary=False)
	bc = lmfit.Parameter('sig_b',value=sigb,min=sig_inst)#,vary=False) sig_inst)minbroad
	yz = lmfit.Parameter('amp_b',value=ampb,min=0.)
	paramsbH.add_many(sl,it,ab,bc,yz,cd,de,ef,fg,gh,hi,rs,st,tu,uv,vw,wy,ij,jk,kl,lm,mn,no,op,pq,qr)

	broadresu = broad_mod.fit(data_cor,paramsbH,x=l)
	lmfit.model.save_modelresult(broadresu, path+'broadone_modelresult.sav')
	with open(path+'fit_broad_result.txt', 'w') as fh:
	    fh.write(broadresu.fit_report())

	# PLOT AND PRINT THE RESULTS 
	refer2 = broad_plot(path,data_head,l,data_cor,meth,trigger,linresu,oneSresu,oneOresu,fin_fit,broadresu,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,std0,std1,z,erz)

	# Ftest
	broad_fit = Ofuncts.funcbroad(l,new_slop,new_intc,
				     broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0'],
				     broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'],
				     broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2'],
				     broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'],
				     broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'],
				     broadresu.values['mu_5'],broadresu.values['sig_5'],broadresu.values['amp_5'],
				     broadresu.values['mu_6'],broadresu.values['sig_6'],broadresu.values['amp_6'],
				     broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])

	pre_x = data_cor[np.where(l>l9)[0][0]-20:np.where(l<l6)[0][-1]+20]-fin_fit[np.where(l>l9)[0][0]-20:np.where(l<l6)[0][-1]+20]
	pre_y = data_cor[np.where(l>l9)[0][0]-20:np.where(l<l6)[0][-1]+20]-broad_fit[np.where(l>l9)[0][0]-20:np.where(l<l6)[0][-1]+20]
	tx, ty = stats.obrientransform(pre_x, pre_y)
	fvalue1, pvalue1 = stats.f_oneway(tx,ty)
	fvalue, pvalue = stats.f_oneway(pre_x,pre_y)
	fvalue2, pvalue2 = stats.levene(pre_x,pre_y)
	fstat = ftest(resu1.chisqr,broadresu.chisqr,resu1.nfree,broadresu.nfree)
	print('')
	print('The probability of a second component (one component vs one + broad Halpha components) in this spectra with the F-test of IDL is: '+str(fstat['p-value']))
	print('The probability of a second component (one component vs one + broad Halpha components) in this spectra with the F-test is: '+str(pvalue))
	print('The probability of a second component (one component vs one + broad Halpha components) in this spectra with the F-test (and O Brien) is: '+str(pvalue1))
	print('The probability of a second component (one component vs one + broad Halpha components) in this spectra with the Levene-test is: '+str(pvalue2))
	print('')
    
    
    
'''
elif trigger == 'N': 


    if os.path.exists(path+'eps_adj'+str(meth)+'_2.txt'): os.remove(path+'eps_adj'+str(meth)+'_2.txt')
    np.savetxt(path+'eps_adj'+str(meth)+'_2.txt',np.c_[stdf_s2/stadev,stdf_s1/stadev,stdf_n2/stadev,stdf_ha/stadev,stdf_n1/stadev,stdf_o1/stadev,stdf_o2/stadev,resu1.chisqr],('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('SII2\tSII1\tNII2\tHa\tNII1\tOI1\tOI2\tChi2'))
    
    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    max2S1 = fin2_fit[np.where(abs(twocompresu.values['mu_0']-l)<0.28)[0][0]] #max(fin2_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    max2S2 = fin2_fit[np.where(abs(twocompresu.values['mu_1']-l)<0.28)[0][0]] #max(fin2_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    max2N1 = fin2_fit[np.where(abs(twocompresu.values['mu_2']-l)<0.28)[0][0]] #max(fin2_fit[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
    max2Ha = fin2_fit[np.where(abs(twocompresu.values['mu_3']-l)<0.28)[0][0]] #max(fin2_fit[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
    max2N2 = fin2_fit[np.where(abs(twocompresu.values['mu_4']-l)<0.28)[0][0]] #max(fin2_fit[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])
    max2O1 = fin2_fit[np.where(abs(twocompresu.values['mu_5']-l)<0.28)[0][0]] #max(fin2_fit[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]])
    max2O2 = fin2_fit[np.where(abs(twocompresu.values['mu_6']-l)<0.28)[0][0]] #max(fin2_fit[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]])
    # two comps
    sig2S2 = pix_to_v*np.sqrt(twocompresu.values['sig_0']**2-sig_inst**2)
    sig20S2 = pix_to_v*np.sqrt(twocompresu.values['sig_20']**2-sig_inst**2)
	
    if tworesu.params['sig_0'].stderr == None: 
	print('Problem determining the errors! First component sigma ')
	esig2S2 = 0.
    elif tworesu.params['sig_0'].stderr != None: 
	esig2S2 = pix_to_v*np.sqrt(twocompresu.values['sig_0']*tworesu.params['sig_0'].stderr)/(np.sqrt(twocompresu.values['sig_0']**2-sig_inst**2))
             
    if tworesu.params['sig_20'].stderr == None:
	print('Problem determining the errors! Second component sigma ')
	esig20S2 = 0.
    elif tworesu.params['sig_20'].stderr != None:
	esig20S2 = pix_to_v*np.sqrt(twocompresu.values['sig_20']*tworesu.params['sig_20'].stderr)/(np.sqrt(twocompresu.values['sig_20']**2-sig_inst**2))
	    
    if meth == 'M1':
	v2S2 = (v_luz*((twocompresu.values['mu_0']-l_SII_2)/l_SII_2))-vsys
	v20S2 = (v_luz*((twocompresu.values['mu_20']-l_SII_2)/l_SII_2))-vsys
	if tworesu.params['mu_0'].stderr == None and tworesu.params['mu_20'].stderr == None: 
	    print('Problem determining the errors! First component ')
	    ev2S2,ev20S2 = 0.,0.
	elif tworesu.params['mu_0'].stderr != None and tworesu.params['mu_20'].stderr != None: 
	    ev2S2 = ((v_luz/l_SII_2)*tworesu.params['mu_0'].stderr)-er_vsys
	    ev20S2 = ((v_luz/l_SII_2)*tworesu.params['mu_20'].stderr)-er_vsys
    elif meth == 'M2':
	v2S2 = (v_luz*((twocompresu.values['mu_5']-l_OI_1)/l_OI_1))-vsys
	v20S2 = (v_luz*((twocompresu.values['mu_25']-l_OI_1)/l_OI_1))-vsys
	if tworesu.params['mu_0'].stderr == None and tworesu.params['mu_20'].stderr == None: 
	    print('Problem determining the errors! First component ')
	    ev2S2,ev20S2 = 0.,0.
	elif tworesu.params['mu_0'].stderr != None and tworesu.params['mu_20'].stderr != None: 
	    ev2S2 = ((v_luz/l_OI_1)*tworesu.params['mu_0'].stderr)-er_vsys
	    ev20S2 = ((v_luz/l_OI_1)*tworesu.params['mu_20'].stderr)-er_vsys

    ################################################ PLOT ######################################################
    plt.close('all')
    # MAIN plot
    fig1   = plt.figure(1,figsize=(10, 9))
    frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(l,data_cor)			     # Initial data
    plt.plot(l,fin2_fit,'r-')
    plt.plot(l,tgaus1,'c--')
    plt.plot(l,tgaus2,'c--')
    plt.plot(l,tgaus3,'c--')
    plt.plot(l,tgaus4,'c--')
    plt.plot(l,tgaus5,'c--')
    plt.plot(l,tgaus6,'c--')
    plt.plot(l,tgaus7,'c--',label='N')
    plt.plot(l,tgaus8,'m--')
    plt.plot(l,tgaus9,'m--')
    plt.plot(l,tgaus10,'m--')
    plt.plot(l,tgaus11,'m--')
    plt.plot(l,tgaus12,'m--')
    plt.plot(l,tgaus13,'m--')
    plt.plot(l,tgaus14,'m--',label='S')
    plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
    textstr = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2S2,ev2S2),
			 r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v20S2,ev20S2),
			 r'$V_{OI_{1-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2O1,ev2O1),
			 r'$V_{OI_{1-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v20O1,ev20O1),
			 r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2S2,esig2S2),
			 r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig20S2,esig20S2),
			 r'$\sigma_{OI_{1-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2O1,esig2O1),
			 r'$\sigma_{OI_{1-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig20O1,esig20O1),
			 r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(max2S2/max2S1),
			 r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(max2Ha)+' $10^{-14}$',
			 r'$\frac{F_{OI_{2}}}{F_{OI_{1}}}$ = '+ '{:.3f}'.format(max2O2/max2O1)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    frame1.text(6350.,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
    plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
    frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
    plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(l[0],l[-1])
    plt.legend(loc='best')

    # RESIDUAL plot
    frame2 = fig1.add_axes((.1,.1,.8,.2))
    plt.plot(l,data_cor-fin2_fit,color='grey')		# Main
    plt.xlabel('Wavelength ($\AA$)',fontsize=14)
    plt.ylabel('Residuals',fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(l[0],l[-1])
    plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
    plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
    plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
    plt.ylim(-(3*stadev)*2,(3*stadev)*2)

    plt.savefig(path+'adj_met'+str(meth)+'_full_2comp.png')
    
    ########################################################################################################################33    
    trigger2 = input('Do the fit needs a broad Halpha component? ("Y"/"N"): ')
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
	ab = lmfit.Parameter('mu_b',value=mub) #6605.,vary=False)
	bc = lmfit.Parameter('sig_b',value=sigb,min=sig_inst) #twocompresu.values['sig_23'])#29.51,vary=False) minbroad
	yz = lmfit.Parameter('amp_b',value=ampb,min=0.)
	paramsbH.add_many(sl,it,rs,st,tu,ab,bc,yz,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,uv,vw,wy,aap,aaq,aar,aas,aat,aau,aaa,aab,aac,aad,aae,aaf,aag,aah,aai,aaj,aak,aal,aam,aan,aao)

	twobroadresu = twobroadcomp_mod.fit(data_cor,paramsbH,x=l)
	lmfit.model.save_modelresult(twobroadresu, path+'broadtwo_meth'+str(meth)+'_modelresult.sav')
	with open(path+'fit_meth'+str(meth)+'_twobroad_result.txt', 'w') as fh:
	    fh.write(twobroadresu.fit_report())

	# PLOT AND PRINT THE RESULTS 
	refer2 = broad_plot(path,data_head,l,data_cor,meth,trigger,linresu,tworesu,fin2_fit,twobroadresu,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,std0,std1,z,erz)

	twobroad_fit = Ofuncts.func2bcom(l,new_slop,new_intc,
					twobroadresu.values['mu_0'],twobroadresu.values['sig_0'],twobroadresu.values['amp_0'],
					twobroadresu.values['mu_1'],twobroadresu.values['sig_1'],twobroadresu.values['amp_1'],
					twobroadresu.values['mu_2'],twobroadresu.values['sig_2'],twobroadresu.values['amp_2'],
					twobroadresu.values['mu_3'],twobroadresu.values['sig_3'],twobroadresu.values['amp_3'],
				        twobroadresu.values['mu_4'],twobroadresu.values['sig_4'],twobroadresu.values['amp_4'],
				        twobroadresu.values['mu_5'],twobroadresu.values['sig_5'],twobroadresu.values['amp_5'],
				        twobroadresu.values['mu_6'],twobroadresu.values['sig_6'],twobroadresu.values['amp_6'],
					twobroadresu.values['mu_20'],twobroadresu.values['sig_20'],twobroadresu.values['amp_20'],
				        twobroadresu.values['mu_21'],twobroadresu.values['sig_21'],twobroadresu.values['amp_21'],
				        twobroadresu.values['mu_22'],twobroadresu.values['sig_22'],twobroadresu.values['amp_22'],
				        twobroadresu.values['mu_23'],twobroadresu.values['sig_23'],twobroadresu.values['amp_23'],
				        twobroadresu.values['mu_24'],twobroadresu.values['sig_24'],twobroadresu.values['amp_24'],
				        twobroadresu.values['mu_25'],twobroadresu.values['sig_25'],twobroadresu.values['amp_25'],
				        twobroadresu.values['mu_26'],twobroadresu.values['sig_26'],twobroadresu.values['amp_26'],
				        twobroadresu.values['mu_b'],twobroadresu.values['sig_b'],twobroadresu.values['amp_b'])
	# F-test		        
	print('')
	pre_x = data_cor[np.where(l>l9)[0][0]-20:np.where(l<l6)[0][-1]+20]-fin2_fit[np.where(l>l9)[0][0]-20:np.where(l<l6)[0][-1]+20]
	pre_y = data_cor[np.where(l>l9)[0][0]-20:np.where(l<l6)[0][-1]+20]-twobroad_fit[np.where(l>l9)[0][0]-20:np.where(l<l6)[0][-1]+20]
	tx, ty = stats.obrientransform(pre_x, pre_y)
	fvalue1, pvalue1 = stats.f_oneway(tx,ty)
	fvalue, pvalue = stats.f_oneway(pre_x,pre_y)
	fvalue2, pvalue2 = stats.levene(pre_x,pre_y)
	fstat = ftest(twocompresu.chisqr,twobroadresu.chisqr,twocompresu.nfree,twobroadresu.nfree)
	print('')
	print('The probability of a third component (two component vs two + broad Halpha components) in this spectra with the F-test of IDL is: '+str(fstat['p-value']))
	print('The probability of a third component (two component vs two + broad Halpha components) in this spectra with the F-test is: '+str(pvalue))
	print('The probability of a third component (two component vs two + broad Halpha components) in this spectra with the F-test (and O Brien) is: '+str(pvalue1))
	print('The probability of a third component (two component vs two + broad Halpha components) in this spectra with the Levene-test is: '+str(pvalue2))
	print('')

else: 
    print('Please use "Y" or "N"')
'''
