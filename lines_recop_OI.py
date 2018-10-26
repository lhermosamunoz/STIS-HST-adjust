'''
This script makes a gaussian fit to the emission lines of AGN spectra
It is needed a path, the spectrum in which the fit is going to be made and the initial estimation of the fit
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import Ofuncts
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
l_Halpha = 6563.
l_NII_1  = 6548.
l_NII_2  = 6584.
l_SII_1  = 6716.
l_SII_2  = 6731.
l_OI_1 = 6300.
l_OI_2 = 6363.

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
    l11 = input('lambda inf for OI 1 (angs)?: ')
    l12 = input('lambda sup for OI 1 (angs)?: ')
    l13 = input('lambda inf for OI 2 (angs)?: ')
    l14 = input('lambda sup for OI 2 (angs)?: ')
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
sig5 = 1.			# OI1
sig25 = 1.5
mu5  = newx6[np.argmax(newy6)]
amp5 = max(newy6)
amp25 = max(newy6)/2.
sig6 = 1.			# OI2
sig26 = 1.5
mu6  = newx7[np.argmax(newy7)]
amp6 = max(newy7)
amp26 = max(newy7)/2.

# Start the parameters for the LINEAR fit
in_slope = 0.
in_intc  = data_cor[0]

# Redefine the lambda zone with the first and last point and the zones in between OI2-NII1 and NII2-SII1
newl = l[1]
zone_O_N = l[np.where(l<l14)[0][-1]+10:np.where(l>l9)[0][0]-10]
zone_N_S = l[np.where(l<l6)[0][-1]+10:np.where(l>l3)[0][0]-10]
newl = np.append(newl,zone_O_N)
newl = np.append(newl,zone_N_S)
newl = np.append(newl,l[-1])
# now we do the same but with the flux data (y vector)
newflux = data_cor[1]
zon_O_N = data_cor[np.where(l<l14)[0][-1]+10:np.where(l>l9)[0][0]-10]
zon_N_S = data_cor[np.where(l<l6)[0][-1]+10:np.where(l>l3)[0][0]-10]
newflux = np.append(newflux,zon_O_N)
newflux = np.append(newflux,zon_N_S)
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
params1 = lmfit.Parameters()
params2 = lmfit.Parameters()

sl = lmfit.Parameter('slop', value=new_slop,vary=False)
it = lmfit.Parameter('intc', value=new_intc,vary=False)
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
meth = input('Which method to be applied? ("S"/"O"/"M1"/"M2"): ')	# Method to fit
if meth == 'S':
    cd = lmfit.Parameter('mu_0', value=mu0)
    de = lmfit.Parameter('sig_0', value=sig0)
    ef = lmfit.Parameter('amp_0', value=amp0,min=0.)
    fg = lmfit.Parameter('mu_1', value=mu1,expr='mu_0*(6716./6731.)')
    gh = lmfit.Parameter('sig_1', value=sig1,expr='sig_0')
    hi = lmfit.Parameter('amp_1', value=amp1,min=0.)
    # second components
    aaa = lmfit.Parameter('mu_20', value=mu0)
    aab = lmfit.Parameter('sig_20', value=sig20)
    aac = lmfit.Parameter('amp_20', value=amp20,min=0.)
    aad = lmfit.Parameter('mu_21', value=mu1,expr='mu_20*(6716./6731.)')
    aae = lmfit.Parameter('sig_21', value=sig21,expr='sig_20')
    aaf = lmfit.Parameter('amp_21', value=amp21,min=0.)
    # Zone of the lines for doing the fit
    x_one = l[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]
    y_one = data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]

elif meth == 'O':
    cd = lmfit.Parameter('mu_0', value=mu5)
    de = lmfit.Parameter('sig_0', value=sig5)
    ef = lmfit.Parameter('amp_0', value=amp5,min=0.)
    fg = lmfit.Parameter('mu_1', value=mu6,expr='mu_0*(6363./6300.)')
    gh = lmfit.Parameter('sig_1', value=sig6,expr='sig_0')
    hi = lmfit.Parameter('amp_1', value=amp6,min=0.,expr='amp_0*(1./3.)')
    # second components
    aaa = lmfit.Parameter('mu_20', value=mu5)
    aab = lmfit.Parameter('sig_20', value=sig25)
    aac = lmfit.Parameter('amp_20', value=amp25,min=0.)
    aad = lmfit.Parameter('mu_21', value=mu6,expr='mu_0*(6363./6300.)')
    aae = lmfit.Parameter('sig_21', value=sig26,expr='sig_20')
    aaf = lmfit.Parameter('amp_21', value=amp26,min=0.,expr='amp_20*(1./3.)')
    # Zone of the lines for doing the fit
    x_one = l[np.where(l>l11)[0][0]:np.where(l<l14)[0][-1]+20]
    y_one = data_cor[np.where(l>l11)[0][0]:np.where(l<l14)[0][-1]+20]

# add a sequence of Parameters
params1.add_many(sl,it,cd,de,ef,fg,gh,hi)
params2.add_many(sl,it,cd,de,ef,fg,gh,hi,aaa,aab,aac,aad,aae,aaf)

###################################################################################################################
# and make the fit using lmfit
oneresu = one_mod.fit(y_one,params1,x=x_one)
tworesu = two_mod.fit(y_one,params2,x=x_one)

##################################### PLOT and PRINT for the SII lines ##################################################
#
# Now we create the individual gaussians in order to plot and print the results for only 1 component
print('				RESULTS OF THE FIT: ')
print('Linear fit equation: {:.5f}*x + {:.5f}'.format(linresu.values['slope'], linresu.values['intc']))
print('')
print('The rest of the results can be displayed all together with two/oneresu.params; the data can be accesed with two/oneresu.values['']')
print('')
print('The chi-square of the fit for 1 gaussian for the reference line is: {:.5f}'.format(oneresu.chisqr))
print('The chi-square of the fit for 2 gaussian for the reference line is: {:.5f}'.format(tworesu.chisqr))
print('')

# Now we create and plot the individual gaussians of the fit
gaus1 = Ofuncts.gaussian(l,oneresu.values['mu_0'],oneresu.values['sig_0'],oneresu.values['amp_0']) 
gaus2 = Ofuncts.gaussian(l,oneresu.values['mu_1'],oneresu.values['sig_1'],oneresu.values['amp_1'])
gaus21 = Ofuncts.gaussian(l,tworesu.values['mu_0'],tworesu.values['sig_0'],tworesu.values['amp_0']) 
gaus22 = Ofuncts.gaussian(l,tworesu.values['mu_1'],tworesu.values['sig_1'],tworesu.values['amp_1'])
gaus23 = Ofuncts.gaussian(l,tworesu.values['mu_20'],tworesu.values['sig_20'],tworesu.values['amp_20'])
gaus24 = Ofuncts.gaussian(l,tworesu.values['mu_21'],tworesu.values['sig_21'],tworesu.values['amp_21'])
onefin_fit = Ofuncts.twogaussian(l,new_slop,new_intc,
				 oneresu.values['mu_0'],oneresu.values['sig_0'],oneresu.values['amp_0'],
				 oneresu.values['mu_1'],oneresu.values['sig_1'],oneresu.values['amp_1'])
twofin_fit = Ofuncts.funcSII2comp(l,new_slop,new_intc,
				 tworesu.values['mu_0'],tworesu.values['sig_0'],tworesu.values['amp_0'],
				 tworesu.values['mu_1'],tworesu.values['sig_1'],tworesu.values['amp_1'],
				 tworesu.values['mu_20'],tworesu.values['sig_20'],tworesu.values['amp_20'],
				 tworesu.values['mu_21'],tworesu.values['sig_21'],tworesu.values['amp_21'])
if meth == 'S':
    # one component
    std_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]]-onefin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]])
    std_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]]-onefin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
    print('		'+str(std_s2)+'< '+str(3*stadev))
    print('		'+str(std_s1)+'< '+str(3*stadev))
    # two components
    std2_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]]-twofin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]])
    std2_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]]-twofin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
    print('		'+str(std2_s2)+'< '+str(3*stadev))
    print('		'+str(std2_s1)+'< '+str(3*stadev))

    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    maxS1 = max(onefin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    maxS2 = max(onefin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    max2S1 = max(twofin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    max2S2 = max(twofin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    # one component
    vS2 = v_luz*((oneresu.values['mu_0']-l_SII_2)/l_SII_2)
    evS2 = (v_luz/l_SII_2)*oneresu.params['mu_0'].stderr
    sigS2 = 47*np.sqrt(oneresu.values['sig_0']**2-sig_inst**2)
    esigS2 = 47*np.sqrt(oneresu.values['sig_0']*oneresu.params['sig_0'].stderr)/(np.sqrt(oneresu.values['sig_0']**2-sig_inst**2))
    # two comps
    v2S2 = v_luz*((tworesu.values['mu_0']-l_SII_2)/l_SII_2)
    v20S2 = v_luz*((tworesu.values['mu_20']-l_SII_2)/l_SII_2)
    ev2S2 = (v_luz/l_SII_2)*tworesu.params['mu_0'].stderr
    ev20S2 = (v_luz/l_SII_2)*tworesu.params['mu_20'].stderr
    sig2S2 = 47*np.sqrt(tworesu.values['sig_0']**2-sig_inst**2)
    sig20S2 = 47*np.sqrt(tworesu.values['sig_20']**2-sig_inst**2)
    esig2S2 = 47*np.sqrt(tworesu.values['sig_0']*tworesu.params['sig_0'].stderr)/(np.sqrt(tworesu.values['sig_0']**2-sig_inst**2))
    esig20S2 = 47*np.sqrt(tworesu.values['sig_20']*tworesu.params['sig_20'].stderr)/(np.sqrt(tworesu.values['sig_20']**2-sig_inst**2))
    textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
		    r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(maxS2)+' $10^{-14}$',
		    r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxS1)+' $10^{-14}$'))
    textstr2 = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2S2,ev2S2),
		    r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v20S2,ev20S2),
		    r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2S2,esig2S2),
		    r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig20S2,esig20S2),
		    r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(max2S2)+' $10^{-14}$',
		    r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(max2S1)+' $10^{-14}$'))

elif meth == 'O':
    # one component
    std_o1 = np.std(data_cor[np.where(l<l11)[0][-1]:np.where(l>l12)[0][0]]-onefin_fit[np.where(l<l11)[0][-1]:np.where(l>l12)[0][0]])
    std_o2 = np.std(data_cor[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]]-onefin_fit[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
    print('		'+str(std_o1)+'< '+str(3*stadev))
    print('		'+str(std_o2)+'< '+str(3*stadev))
    # two components
    std2_o1 = np.std(data_cor[np.where(l<l11)[0][-1]:np.where(l>l12)[0][0]]-twofin_fit[np.where(l<l11)[0][-1]:np.where(l>l12)[0][0]])
    std2_o2 = np.std(data_cor[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]]-twofin_fit[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
    print('		'+str(std2_o1)+'< '+str(3*stadev))
    print('		'+str(std2_o2)+'< '+str(3*stadev))

    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    maxS1 = max(onefin_fit[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]])
    maxS2 = max(onefin_fit[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]])
    max2S1 = max(twofin_fit[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]])
    max2S2 = max(twofin_fit[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]])
    # one component
    vS2 = v_luz*((oneresu.values['mu_0']-l_OI_1)/l_OI_1)
    evS2 = (v_luz/l_OI_1)*oneresu.params['mu_0'].stderr
    sigS2 = 47*np.sqrt(oneresu.values['sig_0']**2-sig_inst**2)
    esigS2 = 47*np.sqrt(oneresu.values['sig_0']*oneresu.params['sig_0'].stderr)/(np.sqrt(oneresu.values['sig_0']**2-sig_inst**2))
    # two comps
    v2S2 = v_luz*((tworesu.values['mu_0']-l_OI_1)/l_OI_1)
    v20S2 = v_luz*((tworesu.values['mu_20']-l_OI_1)/l_OI_1)
    ev2S2 = (v_luz/l_OI_1)*tworesu.params['mu_0'].stderr
    ev20S2 = (v_luz/l_OI_1)*tworesu.params['mu_20'].stderr
    sig2S2 = 47*np.sqrt(tworesu.values['sig_0']**2-sig_inst**2)
    sig20S2 = 47*np.sqrt(tworesu.values['sig_20']**2-sig_inst**2)
    esig2S2 = 47*np.sqrt(tworesu.values['sig_0']*tworesu.params['sig_0'].stderr)/(np.sqrt(tworesu.values['sig_0']**2-sig_inst**2))
    esig20S2 = 47*np.sqrt(tworesu.values['sig_20']*tworesu.params['sig_20'].stderr)/(np.sqrt(tworesu.values['sig_20']**2-sig_inst**2))
    textstr = '\n'.join((r'$V_{OI_{1}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
		    r'$\sigma_{OI_{1}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    r'$F_{OI_{1}}$ = '+ '{:.3f}'.format(maxS2)+' $10^{-14}$',
		    r'$F_{OI_{2}}$ = '+ '{:.3f}'.format(maxS1)+' $10^{-14}$'))
    textstr2 = '\n'.join((r'$V_{OI_{1-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2S2,ev2S2),
		    r'$V_{OI_{1-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v20S2,ev20S2),
		    r'$\sigma_{OI_{1-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2S2,esig2S2),
		    r'$\sigma_{OI_{1-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig20S2,esig20S2),
		    r'$F_{OI_{1}}$ = '+ '{:.3f}'.format(max2S2)+' $10^{-14}$',
		    r'$F_{OI_{2}}$ = '+ '{:.3f}'.format(max2S1)+' $10^{-14}$'))

################################################ PLOT ######################################################
plt.close()
# MAIN plot
fig1   = plt.figure(1,figsize=(10, 9))
frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,onefin_fit,'r--')
plt.plot(l,gaus1,'c--')
plt.plot(l,gaus2,'c--',label='N')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
frame1.text(6850.,oneresu.values['amp_0']+10., textstr, fontsize=12,verticalalignment='top', bbox=props)
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame2 = fig1.add_axes((.1,.1,.8,.2))
plt.plot(l,data_cor-onefin_fit,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
plt.ylim(-2,2)

plt.savefig(path+'adj_met'+str(meth)+'_ref_1comp.png')

#######################################################################################
# Two components in reference line
# MAIN plot
fig2   = plt.figure(2,figsize=(10, 9))
frame3 = fig2.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor)			     # Initial data
plt.plot(l,twofin_fit,'r--')
plt.plot(l,gaus21,'c--')
plt.plot(l,gaus22,'c--',label='N')
plt.plot(l,gaus23,'m--')
plt.plot(l,gaus24,'m--',label='S')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
frame3.text(6850.,tworesu.values['amp_0']+10., textstr2, fontsize=12,verticalalignment='top', bbox=props)
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

frame3.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame4 = fig2.add_axes((.1,.1,.8,.2))
plt.plot(l,data_cor-twofin_fit,color='grey')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
plt.ylim(-2,2)

plt.savefig(path+'adj_met'+str(meth)+'_ref_2comp.png')

##############################################################################################################################################################################
# We make an F-test to see if it is significant the presence of a second component in the lines. 
# As the only possible method here is the S-method due to the fact that there are no O-lines in this spectra, 
# then the method can only be applied to the SII lines (so the wavelength range would be around this two lines)
'''
fvalue, pvalue = stats.f_oneway(data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-onefin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20],data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-twofin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20])
print('')
print('The probability of a second component (one component vs two components) in this spectra is: '+str(pvalue))
print('')
'''

#######################################################################################################################################
# Select if one or two components in the SII lines and then apply to the rest
trigger = input('Is the fit good enough with one component? ("Y"/"N"): ')
'''
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
    kl = lmfit.Parameter('amp_2', value=amp2,min=0.)
    lm = lmfit.Parameter('mu_3', value=mu3,expr='mu_0*(6563./6731.)')
    mn = lmfit.Parameter('sig_3', value=sig3,expr='sig_0')
    no = lmfit.Parameter('amp_3', value=amp3,min=0.)
    op = lmfit.Parameter('mu_4', value=mu4,expr='mu_0*(6548./6731.)')
    pq = lmfit.Parameter('sig_4', value=sig4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value=amp4,min=0.,expr='amp_2*(1./3.)')

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
    stdf_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]]-fin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]])
    stdf_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]]-fin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]])
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
    fig1   = plt.figure(1,figsize=(10, 9))
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
    plt.ylim(-max(data_cor-fin_fit),max(data_cor-fin_fit))

    plt.savefig(path+'adj_metS_full_1comp.png')
    
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
	ab = lmfit.Parameter('mu_b',value=mub)
	bc = lmfit.Parameter('sig_b',value=sigb)
	rs = lmfit.Parameter('amp_b',value=ampb)
	paramsbH.add_many(ab,bc,rs,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr)

    	broadresu = broad_mod.fit(data_cor,paramsbH,x=l)

        ################################## Calculate gaussians and final fit #######################################
	# Now we create and plot the individual gaussians of the fit
	bgaus1 = gaussian(l,broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0']) 
	bgaus2 = gaussian(l,broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'])
    	bgaus3 = gaussian(l,broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2']) 
	bgaus4 = gaussian(l,broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'])
    	bgaus5 = gaussian(l,broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'])
    	bgaus6 = gaussian(l,broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])
    	broad_fit = funcbroad(l,broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0'],
			      broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'],
			      broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2'],
			      broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'],
			      broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'],
			      broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])
	
    	stdb_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]]-broad_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]])
    	stdb_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]]-broad_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]])
	stdb_n2 = np.std(data_cor[np.where(l<l5)[0][-1]:np.where(l>l6)[0][0]]-broad_fit[np.where(l<l5)[0][-1]:np.where(l>l6)[0][0]])
	stdb_ha = np.std(data_cor[np.where(l<l7)[0][-1]:np.where(l>l8)[0][0]]-broad_fit[np.where(l<l7)[0][-1]:np.where(l>l8)[0][0]])
	stdb_n1 = np.std(data_cor[np.where(l<l9)[0][-1]:np.where(l>l10)[0][0]]-broad_fit[np.where(l<l9)[0][-1]:np.where(l>l10)[0][0]])
    	print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component + Ha is... ')
    	print('		For SII2: '+str(stdb_s2)+'< '+str(3*stadev))
    	print('		For SII1: '+str(stdb_s1)+'< '+str(3*stadev))
    	print('		For NII2: '+str(stdb_n2)+'< '+str(3*stadev))
    	print('		For Halp: '+str(stdb_ha)+'< '+str(3*stadev))
    	print('		For SII1: '+str(stdb_n1)+'< '+str(3*stadev))

   	# We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    	maxbS1 = max(broad_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    	maxbS2 = max(broad_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    	maxbN1 = max(broad_fit[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
    	maxbHa = max(broad_fit[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
    	maxbN2 = max(broad_fit[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])
        # two comps
        vbS2 = v_luz*((broadresu.values['mu_0']-l_SII_2)/l_SII_2)
        vb0S2 = v_luz*((broadresu.values['mu_b']-l_Halpha)/l_Halpha)
        evbS2 = (v_luz/l_SII_2)*SIIresu.params['mu_0'].stderr
        evb0S2 = (v_luz/l_SII_2)*broadresu.params['mu_b'].stderr
        sigbS2 = 47*np.sqrt(broadresu.values['sig_0']**2-sig_inst**2)
        sigb0S2 = 47*np.sqrt(broadresu.values['sig_b']**2-sig_inst**2)
        esigbS2 = 47*np.sqrt(broadresu.values['sig_0']*SIIresu.params['sig_0'].stderr)/(np.sqrt(broadresu.values['sig_0']**2-sig_inst**2))
        esigb0S2 = 47*np.sqrt(broadresu.values['sig_b']*broadresu.params['sig_b'].stderr)/(np.sqrt(broadresu.values['sig_b']**2-sig_inst**2))

	#############################################################################################################
	# We make an F-test to see if it is significant the presence of a broad component in the lines. 
	fbvalue, pbvalue = stats.f_oneway(data_cor[np.where(l<l9)[0][-1]:np.where(l>l6)[0][0]]-fin_fit[np.where(l<l9)[0][-1]:np.where(l>l6)[0][0]],data_cor[np.where(l<l9)[0][-1]:np.where(l>l6)[0][0]]-broad_fit[np.where(l<l9)[0][-1]:np.where(l>l6)[0][0]])
	print('')
	print('The probability of a second component (one component vs one + broad Halpha components) in this spectra is: '+str(pbvalue))
	print('')

   	################################################ PLOT ######################################################
    	plt.close('all')
    	# MAIN plot
    	fig1   = plt.figure(1,figsize=(10, 9))
    	frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    	plt.plot(l,data_cor)			     # Initial data
    	plt.plot(l,broad_fit,'r--')
    	plt.plot(l,bgaus1,'c--')
    	plt.plot(l,bgaus2,'c--')
    	plt.plot(l,bgaus3,'c--')
    	plt.plot(l,bgaus4,'c--')
    	plt.plot(l,bgaus5,'c--',label='N')
    	plt.plot(l,bgaus6,'m--',label='B')
    	plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
    	textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vbS2,evbS2),
			r'$V_{SII_{2-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(vb0S2,evb0S2),
		    	r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigbS2,esigbS2),
		    	r'$\sigma_{SII_{2-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(sigb0S2,esigb0S2),
		    	r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(maxbS2)+' $10^{-14}$',
		    	r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxbS1)+' $10^{-14}$',
		    	r'$F_{NII_{2}}$ = '+ '{:.3f}'.format(maxbN2)+' $10^{-14}$',
		    	r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxbHa)+' $10^{-14}$',
		    	r'$F_{NII_{1}}$ = '+ '{:.3f}'.format(maxbN1)+' $10^{-14}$'))
    	props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    	frame1.text(6850.,broadresu.values['amp_0']+12., textstr, fontsize=12,verticalalignment='top', bbox=props)
    	plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
    	frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
    	plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
    	plt.tick_params(axis='both', labelsize=12)
    	plt.xlim(l[0],l[-1])
    	plt.legend(loc='best')

    	# RESIDUAL plot
    	frame2 = fig1.add_axes((.1,.1,.8,.2))
    	plt.plot(l,data_cor-broad_fit,color='grey')		# Main
    	plt.xlabel('Wavelength ($\AA$)',fontsize=14)
    	plt.ylabel('Residuals',fontsize=14)
    	plt.tick_params(axis='both', labelsize=12)
    	plt.xlim(l[0],l[-1])
    	plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
    	plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
    	plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
    	plt.ylim(-max(data_cor-broad_fit),max(data_cor-broad_fit))

    	plt.savefig(path+'adj_metS_full_1comp_broadH.png')


elif trigger == 'N':
    sig22 = 10.				# NII2
    amp22 = max(newy3)/2.
    sig23 = 10.				# Halpha
    amp23 = max(newy4)/2.
    sig24 = 10.				# NII1
    amp24 = max(newy5)/2.
    # Now we define the initial guesses and the constraints
    params2c = lmfit.Parameters()
    cd = lmfit.Parameter('mu_0', value=twoSIIresu.values["mu_0"],vary=False)
    de = lmfit.Parameter('sig_0', value=twoSIIresu.values["sig_0"],vary=False)
    ef = lmfit.Parameter('amp_0', value=twoSIIresu.values["amp_0"],vary=False)
    fg = lmfit.Parameter('mu_1', value=twoSIIresu.values["mu_1"],vary=False)
    gh = lmfit.Parameter('sig_1', value=twoSIIresu.values["sig_1"],vary=False)
    hi = lmfit.Parameter('amp_1', value=twoSIIresu.values["amp_1"],vary=False)
    ij = lmfit.Parameter('mu_2', value=mu2,expr='mu_0*(6584./6731.)')
    jk = lmfit.Parameter('sig_2', value=sig2,expr='sig_0')
    kl = lmfit.Parameter('amp_2', value=amp2,min=0.)
    lm = lmfit.Parameter('mu_3', value=mu3,expr='mu_0*(6563./6731.)')
    mn = lmfit.Parameter('sig_3', value=sig3,expr='sig_0')
    no = lmfit.Parameter('amp_3', value=amp3,min=0.)
    op = lmfit.Parameter('mu_4', value=mu4,expr='mu_0*(6548./6731.)')
    pq = lmfit.Parameter('sig_4', value=sig4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value=amp4,expr='amp_2*(1./3.)')
    # second components
    aaa = lmfit.Parameter('mu_20', value=twoSIIresu.values["mu_20"],vary=False)
    aab = lmfit.Parameter('sig_20', value=twoSIIresu.values["sig_20"],vary=False)
    aac = lmfit.Parameter('amp_20', value=twoSIIresu.values["amp_20"],vary=False)
    aad = lmfit.Parameter('mu_21', value=twoSIIresu.values["mu_21"],vary=False)
    aae = lmfit.Parameter('sig_21', value=twoSIIresu.values["sig_21"],vary=False)
    aaf = lmfit.Parameter('amp_21', value=twoSIIresu.values["amp_21"],vary=False)
    aag = lmfit.Parameter('mu_22', value=mu2,expr='mu_20*(6584./6731.)')
    aah = lmfit.Parameter('sig_22', value=sig22,expr='sig_20')
    aai = lmfit.Parameter('amp_22', value=amp22,min=0.)
    aaj = lmfit.Parameter('mu_23', value=mu3,expr='mu_20*(6563./6731.)')
    aak = lmfit.Parameter('sig_23', value=sig23,expr='sig_20')
    aal = lmfit.Parameter('amp_23', value=amp23,min=0.)
    aam = lmfit.Parameter('mu_24', value=mu4,expr='mu_20*(6548./6731.)')
    aan = lmfit.Parameter('sig_24', value=sig24,expr='sig_20')
    aao = lmfit.Parameter('amp_24', value=amp24,min=0.,expr='amp_22*(1./3.)')
    params2c.add_many(cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,aaa,aab,aac,aad,aae,aaf,aag,aah,aai,aaj,aak,aal,aam,aan,aao)

    twocompresu = twocomp_mod.fit(data_cor,params2c,x=l)

    ################################## Calculate gaussians and final fit #######################################
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
    fin2_fit = func2com(l,twocompresu.values['mu_0'],twocompresu.values['sig_0'],twocompresu.values['amp_0'],
		       twocompresu.values['mu_1'],twocompresu.values['sig_1'],twocompresu.values['amp_1'],
		       twocompresu.values['mu_2'],twocompresu.values['sig_2'],twocompresu.values['amp_2'],
		       twocompresu.values['mu_3'],twocompresu.values['sig_3'],twocompresu.values['amp_3'],
		       twocompresu.values['mu_4'],twocompresu.values['sig_4'],twocompresu.values['amp_4'],
		       twocompresu.values['mu_20'],twocompresu.values['sig_20'],twocompresu.values['amp_20'],
		       twocompresu.values['mu_21'],twocompresu.values['sig_21'],twocompresu.values['amp_21'],
		       twocompresu.values['mu_22'],twocompresu.values['sig_22'],twocompresu.values['amp_22'],
		       twocompresu.values['mu_23'],twocompresu.values['sig_23'],twocompresu.values['amp_23'],
		       twocompresu.values['mu_24'],twocompresu.values['sig_24'],twocompresu.values['amp_24'],)

    # two components
    stdf2_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]]-fin2_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]])
    stdf2_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]]-fin2_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
    print('		'+str(stdf2_s2)+'< '+str(3*stadev))
    print('		'+str(stdf2_s1)+'< '+str(3*stadev))

    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    maxb2S1 = max(fin2_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    maxb2S2 = max(fin2_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    maxb2N1 = max(fin2_fit[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
    maxb2Ha = max(fin2_fit[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
    maxb2N2 = max(fin2_fit[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])
    # two comps
    vb2S2 = v_luz*((twocompresu.values['mu_0']-l_SII_2)/l_SII_2)
    vb20S2 = v_luz*((twocompresu.values['mu_20']-l_SII_2)/l_SII_2)
    vb20S2 = v_luz*((twocompresu.values['mu_20']-l_SII_2)/l_SII_2)
    evb2S2 = (v_luz/l_SII_2)*twoSIIresu.params['mu_0'].stderr
    evb20S2 = (v_luz/l_SII_2)*twoSIIresu.params['mu_20'].stderr
    sigb2S2 = 47*np.sqrt(twocompresu.values['sig_0']**2-sig_inst**2)
    sigb20S2 = 47*np.sqrt(twocompresu.values['sig_20']**2-sig_inst**2)
    esigb2S2 = 47*np.sqrt(twocompresu.values['sig_0']*twoSIIresu.params['sig_0'].stderr)/(np.sqrt(twocompresu.values['sig_0']**2-sig_inst**2))
    esigb20S2 = 47*np.sqrt(twocompresu.values['sig_20']*twoSIIresu.params['sig_20'].stderr)/(np.sqrt(twocompresu.values['sig_20']**2-sig_inst**2))

    ################################################ PLOT ######################################################
    plt.close('all')
    # MAIN plot
    fig1   = plt.figure(1,figsize=(10, 9))
    frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(l,data_cor)			     # Initial data
    plt.plot(l,fin2_fit,'r--')
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
    textstr = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vb2S2,evb2S2),
		    r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vb20S2,evb20S2),
		    r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigb2S2,esigb2S2),
		    r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigb20S2,esigb20S2),
		    r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(maxb2S2)+' $10^{-14}$',
		    r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxb2S1)+' $10^{-14}$',
		    r'$F_{NII_{2}}$ = '+ '{:.3f}'.format(maxb2N2)+' $10^{-14}$',
		    r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxb2Ha)+' $10^{-14}$',
		    r'$F_{NII_{1}}$ = '+ '{:.3f}'.format(maxb2N1)+' $10^{-14}$'))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    frame1.text(6850.,twocompresu.values['amp_0']+13.5, textstr, fontsize=12,verticalalignment='top', bbox=props)
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
    plt.ylim(-max(data_cor-fin2_fit),max(data_cor-fin2_fit))

    plt.savefig(path+'adj_metS_full_2comp.png')

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
	ab = lmfit.Parameter('mu_b',value=mub)
	bc = lmfit.Parameter('sig_b',value=sigb)
	rs = lmfit.Parameter('amp_b',value=ampb)
	paramsbH.add_many(ab,bc,rs,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,aaa,aab,aac,aad,aae,aaf,aag,aah,aai,aaj,aak,aal,aam,aan,aao)

    	twobroadresu = twobroadcomp_mod.fit(data_cor,paramsbH,x=l)

        ################################## Calculate gaussians and final fit #######################################
	# Now we create and plot the individual gaussians of the fit
	b2gaus1 = gaussian(l,twobroadresu.values['mu_0'],twobroadresu.values['sig_0'],twobroadresu.values['amp_0']) 
	b2gaus2 = gaussian(l,twobroadresu.values['mu_1'],twobroadresu.values['sig_1'],twobroadresu.values['amp_1'])
    	b2gaus3 = gaussian(l,twobroadresu.values['mu_2'],twobroadresu.values['sig_2'],twobroadresu.values['amp_2']) 
	b2gaus4 = gaussian(l,twobroadresu.values['mu_3'],twobroadresu.values['sig_3'],twobroadresu.values['amp_3'])
    	b2gaus5 = gaussian(l,twobroadresu.values['mu_4'],twobroadresu.values['sig_4'],twobroadresu.values['amp_4'])
	b2gaus6 = gaussian(l,twobroadresu.values['mu_20'],twobroadresu.values['sig_20'],twobroadresu.values['amp_20']) 
	b2gaus7 = gaussian(l,twobroadresu.values['mu_21'],twobroadresu.values['sig_21'],twobroadresu.values['amp_21'])
    	b2gaus8 = gaussian(l,twobroadresu.values['mu_22'],twobroadresu.values['sig_22'],twobroadresu.values['amp_22']) 
	b2gaus9 = gaussian(l,twobroadresu.values['mu_23'],twobroadresu.values['sig_23'],twobroadresu.values['amp_23'])
    	b2gaus10 = gaussian(l,twobroadresu.values['mu_24'],twobroadresu.values['sig_24'],twobroadresu.values['amp_24'])
    	b2gausb = gaussian(l,twobroadresu.values['mu_b'],twobroadresu.values['sig_b'],twobroadresu.values['amp_b'])
    	twobroad_fit = func2bcom(l,twobroadresu.values['mu_0'],twobroadresu.values['sig_0'],twobroadresu.values['amp_0'],
			        twobroadresu.values['mu_1'],twobroadresu.values['sig_1'],twobroadresu.values['amp_1'],
			        twobroadresu.values['mu_2'],twobroadresu.values['sig_2'],twobroadresu.values['amp_2'],
			        twobroadresu.values['mu_3'],twobroadresu.values['sig_3'],twobroadresu.values['amp_3'],
			        twobroadresu.values['mu_4'],twobroadresu.values['sig_4'],twobroadresu.values['amp_4'],
				twobroadresu.values['mu_20'],twobroadresu.values['sig_20'],twobroadresu.values['amp_20'],
			        twobroadresu.values['mu_21'],twobroadresu.values['sig_21'],twobroadresu.values['amp_21'],
			        twobroadresu.values['mu_22'],twobroadresu.values['sig_22'],twobroadresu.values['amp_22'],
			        twobroadresu.values['mu_23'],twobroadresu.values['sig_23'],twobroadresu.values['amp_23'],
			        twobroadresu.values['mu_24'],twobroadresu.values['sig_24'],twobroadresu.values['amp_24'],
			        twobroadresu.values['mu_b'],twobroadresu.values['sig_b'],twobroadresu.values['amp_b'])
	
    	stdb2_s2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]]-twobroad_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]])
    	stdb2_s1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]]-twobroad_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]])
	stdb2_n2 = np.std(data_cor[np.where(l<l5)[0][-1]:np.where(l>l6)[0][0]]-twobroad_fit[np.where(l<l5)[0][-1]:np.where(l>l6)[0][0]])
	stdb2_ha = np.std(data_cor[np.where(l<l7)[0][-1]:np.where(l>l8)[0][0]]-twobroad_fit[np.where(l<l7)[0][-1]:np.where(l>l8)[0][0]])
	stdb2_n1 = np.std(data_cor[np.where(l<l9)[0][-1]:np.where(l>l10)[0][0]]-twobroad_fit[np.where(l<l9)[0][-1]:np.where(l>l10)[0][0]])
    	print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component + Ha is... ')
    	print('		For SII2: '+str(stdb2_s2)+'< '+str(3*stadev))
    	print('		For SII1: '+str(stdb2_s1)+'< '+str(3*stadev))
    	print('		For NII2: '+str(stdb2_n2)+'< '+str(3*stadev))
    	print('		For Halp: '+str(stdb2_ha)+'< '+str(3*stadev))
    	print('		For SII1: '+str(stdb2_n1)+'< '+str(3*stadev))

   	# We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    	maxfb2S1 = max(twobroad_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
    	maxfb2S2 = max(twobroad_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
    	maxfb2N1 = max(twobroad_fit[np.where(l>l9)[0][0]:np.where(l<l10)[0][-1]])
    	maxfb2Ha = max(twobroad_fit[np.where(l>l7)[0][0]:np.where(l<l8)[0][-1]])
    	maxfb2N2 = max(twobroad_fit[np.where(l>l5)[0][0]:np.where(l<l6)[0][-1]])
        # two comps
        vfbS2 = v_luz*((twobroadresu.values['mu_0']-l_SII_2)/l_SII_2)
        vfb2S2 = v_luz*((twobroadresu.values['mu_20']-l_SII_2)/l_SII_2)
        vfb0S2 = v_luz*((twobroadresu.values['mu_b']-l_Halpha)/l_Halpha)
        evfbS2 = (v_luz/l_SII_2)*twoSIIresu.params['mu_0'].stderr
        evfb2S2 = (v_luz/l_SII_2)*twoSIIresu.params['mu_20'].stderr
        evfb0S2 = (v_luz/l_SII_2)*twobroadresu.params['mu_b'].stderr
        sigfbS2 = 47*np.sqrt(twobroadresu.values['sig_0']**2-sig_inst**2)
        sigfb2S2 = 47*np.sqrt(twobroadresu.values['sig_20']**2-sig_inst**2)
        sigfb0S2 = 47*np.sqrt(twobroadresu.values['sig_b']**2-sig_inst**2)
        esigfbS2 = 47*np.sqrt(twobroadresu.values['sig_0']*twoSIIresu.params['sig_0'].stderr)/(np.sqrt(twobroadresu.values['sig_0']**2-sig_inst**2))
        esigfb2S2 = 47*np.sqrt(twobroadresu.values['sig_20']*twoSIIresu.params['sig_20'].stderr)/(np.sqrt(twobroadresu.values['sig_20']**2-sig_inst**2))
        esigfb0S2 = 47*np.sqrt(twobroadresu.values['sig_b']*twobroadresu.params['sig_b'].stderr)/(np.sqrt(twobroadresu.values['sig_b']**2-sig_inst**2))

	#############################################################################################################
	# We make an F-test to see if it is significant the presence of a broad component in the lines. 
	fb2value, pb2value = stats.f_oneway(data_cor[np.where(l<l9)[0][-1]:np.where(l>l6)[0][0]]-fin2_fit[np.where(l<l9)[0][-1]:np.where(l>l6)[0][0]],data_cor[np.where(l<l9)[0][-1]:np.where(l>l6)[0][0]]-twobroad_fit[np.where(l<l9)[0][-1]:np.where(l>l6)[0][0]])
	print('')
	print('The probability of a third component (two component vs two + broad Halpha components) in this spectra is: '+str(pb2value))
	print('')
   	################################################ PLOT ######################################################
    	plt.close('all')
    	# MAIN plot
    	fig1   = plt.figure(1,figsize=(11, 10))
    	frame1 = fig1.add_axes((.1,.3,.8,.6)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    	plt.plot(l,data_cor)			     # Initial data
    	plt.plot(l,twobroad_fit,'r--')
    	plt.plot(l,b2gaus1,'c--')
    	plt.plot(l,b2gaus2,'c--')
    	plt.plot(l,b2gaus3,'c--')
    	plt.plot(l,b2gaus4,'c--')
    	plt.plot(l,b2gaus5,'c--',label='N')
    	plt.plot(l,b2gaus6,'y--')
    	plt.plot(l,b2gaus7,'y--')
    	plt.plot(l,b2gaus8,'y--')
    	plt.plot(l,b2gaus9,'y--')
    	plt.plot(l,b2gaus10,'y--',label='S')
    	plt.plot(l,b2gausb,'m--',label='B')
    	plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'k-.',label='Linear fit')
    	textstr = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vfbS2,evfbS2),
			r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vfb2S2,evfb2S2),
			r'$V_{SII_{2-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(vfb0S2,evfb0S2),
		    	r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigfbS2,esigfbS2),
		    	r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigfb2S2,esigfb2S2),
		    	r'$\sigma_{SII_{2-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(sigfb0S2,esigfb0S2),
		    	r'$F_{SII_{2}}$ = '+ '{:.3f}'.format(maxfb2S2)+' $10^{-14}$',
		    	r'$F_{SII_{1}}$ = '+ '{:.3f}'.format(maxfb2S1)+' $10^{-14}$',
		    	r'$F_{NII_{2}}$ = '+ '{:.3f}'.format(maxfb2N2)+' $10^{-14}$',
		    	r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxfb2Ha)+' $10^{-14}$',
		    	r'$F_{NII_{1}}$ = '+ '{:.3f}'.format(maxfb2N1)+' $10^{-14}$'))
    	props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    	frame1.text(6850.,twobroadresu.values['amp_0']+12., textstr, fontsize=12,verticalalignment='top', bbox=props)
    	plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
    	frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
    	plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
    	plt.tick_params(axis='both', labelsize=12)
    	plt.xlim(l[0],l[-1])
    	plt.legend(loc='best')

    	# RESIDUAL plot
    	frame2 = fig1.add_axes((.1,.1,.8,.2))
    	plt.plot(l,data_cor-twobroad_fit,color='grey')		# Main
    	plt.xlabel('Wavelength ($\AA$)',fontsize=14)
    	plt.ylabel('Residuals',fontsize=14)
    	plt.tick_params(axis='both', labelsize=12)
    	plt.xlim(l[0],l[-1])
    	plt.plot(l,np.zeros(len(l)),'k--')         	# Line around zero
    	plt.plot(l,np.zeros(len(l))+3*stadev,'k--')	# 3 sigma upper limit
    	plt.plot(l,np.zeros(len(l))-3*stadev,'k--') 	# 3 sigma down limit
    	plt.ylim(-max(data_cor-twobroad_fit),max(data_cor-twobroad_fit))
    	plt.savefig(path+'adj_metS_full_2comp_broadH.png')

else: 
    print('Please use "Y" or "N"')
'''

