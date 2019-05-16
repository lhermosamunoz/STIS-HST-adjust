'''
This script makes a gaussian fit to the emission lines of AGN spectra
It is needed a path, the spectrum in which the fit is going to be made and the initial estimation of the fit
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import noOfuncts
import lmfit
import scipy.stats as stats
from PyAstronomy.pyasl import ftest
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
l_OI_2 = 6363.776

# Constants and STIS parameters
v_luz = 299792.458 # km/s
plate_scale = data_head['PLATESC']
fwhm = 2*np.sqrt(2*np.log(2)) # por sigma
pix_to_v = 47	# km/s
if plate_scale == 0.05078:
    siginst = 1.1	# A if binning 1x1 // 2.2 if binning 1x2
    sig_inst = siginst/fwhm	# considering a gaussian, same units as in the fit
    ang_to_pix = 0.554
#    pix_to_v = 25	# km/s
    minbroad = 24.
elif plate_scale == 0.10156:
    siginst = 2.2	# A if binning 1x1 // 2.2 if binning 1x2
    sig_inst = siginst/fwhm	# considering a gaussian, same units as in the fit
    ang_to_pix = 1.108
    pix_to_v = 47	# km/s
    minbroad = 13.#2.83402

# Now redefine the zone to fit
data_cor = data[2:-2]*10**14
l_init = l[2:-2]

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
    z = t[-2,]
    erz = t[-1,]

############################  Redshift correction #############################################
# Redshift correction based on SII2
red_lambda_cor_SII2 = (l_SII_2*(1+z))-l_SII_2
l = l_init-red_lambda_cor_SII2
####################################################################################
newx1 = l[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]+1]		# SII2
newy1 = data_cor[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]+1]
newx2 = l[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]+1]		# SII1
newy2 = data_cor[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]+1]
newx3 = l[np.where(l_init>l5)[0][0]:np.where(l_init<l6)[0][-1]+1]		# NII2
newy3 = data_cor[np.where(l_init>l5)[0][0]:np.where(l_init<l6)[0][-1]+1]
newx4 = l[np.where(l_init>l7)[0][0]:np.where(l_init<l8)[0][-1]+1]		# Halpha
newy4 = data_cor[np.where(l_init>l7)[0][0]:np.where(l_init<l8)[0][-1]+1]
newx5 = l[np.where(l_init>l9)[0][0]:np.where(l_init<l10)[0][-1]+1]	# NII1
newy5 = data_cor[np.where(l_init>l9)[0][0]:np.where(l_init<l10)[0][-1]+1]

# Initial guesses of the fitting parameters
sig0 = 2.3			# SII2
sig20 = 3.
mu0  = newx1[np.argmax(newy1)]
amp0 = max(newy1)
amp20 = max(newy1)/2.
sig1 = 2.3			# SII1
sig21 = 3.
mu1 = newx2[np.argmax(newy2)]
amp1 = max(newy2)
amp21 = max(newy2)/2.

# Start the parameters for the LINEAR fit
in_slope = 0.
in_intc  = data_cor[0]

# Redefine the lambda zone with the first and last point and the zones in between NII2-SII1 and SII2-final
newl = l[1]
l11 = float(input('Aprox max wavelength of the spectra?: '))
if l11<6900.:
    zone_extra = l[np.where(l_init<6400.)[0][-1]+10:np.where(l_init>l10)[0][0]-50]
    newl = np.append(newl,zone_extra)
zone_S_fin = l[np.where(l_init<l2)[0][-1]+10:np.where(l_init>l11)[0][0]]
zone_N_S = l[np.where(l_init<l6)[0][-1]+10:np.where(l_init>l3)[0][0]-10]
newl = np.append(newl,zone_N_S)
newl = np.append(newl,zone_S_fin)
newl = np.append(newl,l[-1])
# now we do the same but with the flux data (y vector)
newflux = data_cor[1]
if l11<6900.:
    zone_extra = data_cor[np.where(l_init<6400.)[0][-1]+10:np.where(l_init>l10)[0][0]-50]
    newflux = np.append(newflux,zone_extra)
zon_S_fin = data_cor[np.where(l_init<l2)[0][-1]+10:np.where(l_init>l11)[0][0]]
zon_N_S = data_cor[np.where(l_init<l6)[0][-1]+10:np.where(l_init>l3)[0][0]-10]
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
std0 = np.where(l_init>float(input('lim inf for determining the stddev of the continuum (angs)?: ')))[0][0]
std1 = np.where(l_init<float(input('lim sup for determining the stddev of the continuum (angs)?: ')))[0][-1]
stadev = np.std(data_cor[std0:std1])

###################################### Start the fit and the MODEL ######################################################
#
# First we have to initialise the model in the SII lines by doing
lin_mod = lmfit.Model(noOfuncts.linear)
sII_mod = lmfit.Model(noOfuncts.twogaussian)
twosII_mod = lmfit.Model(noOfuncts.funcSII2comp)
# and initialise the model in the whole spectra for several different models
comp_mod = lmfit.Model(noOfuncts.funcgauslin)
broad_mod = lmfit.Model(noOfuncts.funcbroad)
twocomp_mod = lmfit.Model(noOfuncts.func2com)
twobroadcomp_mod = lmfit.Model(noOfuncts.func2bcom)

# We make the linear fit only with some windows of the spectra, and calculate the line to introduce it in the formula
linresu  = lin_mod.fit(newflux,slope=in_slope,intc=in_intc,x=newl)
new_slop = linresu.values['slope']
new_intc = linresu.values['intc']
lin_data_fin = (linresu.values['slope']*l+linresu.values['intc'])

# Now we define the initial guesses and the constraints
paramsSII = lmfit.Parameters()
params2SII = lmfit.Parameters()
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
print('The method to be applied is S-method as there are not OI lines available!')	# Method to fit
sl = lmfit.Parameter('slop', value = new_slop,vary = False)
it = lmfit.Parameter('intc', value = new_intc,vary = False)
cd = lmfit.Parameter('mu_0', value = mu0)
de = lmfit.Parameter('sig_0', value = sig0, min=sig_inst)#, max=12.7)
ef = lmfit.Parameter('amp_0', value = amp0,min=0.)
fg = lmfit.Parameter('mu_1', value = mu1,expr='mu_0*(6716.44/6730.82)')
gh = lmfit.Parameter('sig_1', value = sig1,expr='sig_0')
hi = lmfit.Parameter('amp_1', value = amp1,min=0.)

# second components
#de2 = lmfit.Parameter('sig_0', value = sig0,min=sig_inst,max=12.7)
aaa = lmfit.Parameter('mu_20', value = mu0)
aab = lmfit.Parameter('sig_20', value = sig20,min=sig_inst)#,max=12.7),max=minbroad
aac = lmfit.Parameter('amp_20', value = amp20,min=0.)
aad = lmfit.Parameter('mu_21', value = mu1,expr='mu_20*(6716.44/6730.82)')
aae = lmfit.Parameter('sig_21', value = sig21,expr='sig_20')
aaf = lmfit.Parameter('amp_21', value = amp21,min=0.)

# add a sequence of Parameters
paramsSII.add_many(sl,it,cd,de,ef,fg,gh,hi)
params2SII.add_many(sl,it,cd,de,ef,fg,gh,hi,aaa,aab,aac,aad,aae,aaf)

########################################################################################################################
# and make the fit using lmfit
SIIresu = sII_mod.fit(data_cor[np.where(l_init>l3)[0][0]-20:np.where(l_init<l2)[0][-1]+20],paramsSII,x=l[np.where(l_init>l3)[0][0]-20:np.where(l_init<l2)[0][-1]+20])
twoSIIresu = twosII_mod.fit(data_cor[np.where(l_init>l3)[0][0]-20:np.where(l_init<l2)[0][-1]+20],params2SII,x=l[np.where(l_init>l3)[0][0]-20:np.where(l_init<l2)[0][-1]+20])

lmfit.model.save_modelresult(SIIresu, path+'SII_modelresult.sav')
lmfit.model.save_modelresult(twoSIIresu, path+'SII_twocomps_modelresult.sav')
with open(path+'fitSII_result.txt', 'w') as fh:
    fh.write(SIIresu.fit_report())
with open(path+'fit_twoSII_result.txt', 'w') as fh:
    fh.write(twoSIIresu.fit_report())

##################################### PLOT and PRINT for the SII lines #################################################
#
# Now we create the individual gaussians in order to plot and print the results for only 1 component
print('				RESULTS OF THE FIT: ')
print('Linear fit equation: {:.5f}*x + {:.5f}'.format(linresu.values['slope'], linresu.values['intc']))
print('')
print('The rest of the results can be displayed all together with two/SIIresu.params; the data can be accesed with two/SIIresu.values['']')
print('')
print('The chi-square of the fit for 1 gaussian for SII is: {:.5f}'.format(SIIresu.chisqr))
print('The chi-square of the fit for 2 gaussian for SII is: {:.5f}'.format(twoSIIresu.chisqr))
print('')

# Now we create and plot the individual gaussians of the fit
gaus1 = noOfuncts.gaussian(l,SIIresu.values['mu_0'],SIIresu.values['sig_0'],SIIresu.values['amp_0']) 
gaus2 = noOfuncts.gaussian(l,SIIresu.values['mu_1'],SIIresu.values['sig_1'],SIIresu.values['amp_1'])
gaus21 = noOfuncts.gaussian(l,twoSIIresu.values['mu_0'],twoSIIresu.values['sig_0'],twoSIIresu.values['amp_0']) 
gaus22 = noOfuncts.gaussian(l,twoSIIresu.values['mu_1'],twoSIIresu.values['sig_1'],twoSIIresu.values['amp_1'])
gaus23 = noOfuncts.gaussian(l,twoSIIresu.values['mu_20'],twoSIIresu.values['sig_20'],twoSIIresu.values['amp_20'])
gaus24 = noOfuncts.gaussian(l,twoSIIresu.values['mu_21'],twoSIIresu.values['sig_21'],twoSIIresu.values['amp_21'])
SIIfin_fit = noOfuncts.twogaussian(l,new_slop,new_intc,
				  SIIresu.values['mu_0'],SIIresu.values['sig_0'],SIIresu.values['amp_0'],
			 	  SIIresu.values['mu_1'],SIIresu.values['sig_1'],SIIresu.values['amp_1'])
SII2fin_fit = noOfuncts.funcSII2comp(l,new_slop,new_intc,
				     twoSIIresu.values['mu_0'],twoSIIresu.values['sig_0'],twoSIIresu.values['amp_0'],
				     twoSIIresu.values['mu_1'],twoSIIresu.values['sig_1'],twoSIIresu.values['amp_1'],
				     twoSIIresu.values['mu_20'],twoSIIresu.values['sig_20'],twoSIIresu.values['amp_20'],
				     twoSIIresu.values['mu_21'],twoSIIresu.values['sig_21'],twoSIIresu.values['amp_21'])

# one component
std_s2 = np.std(data_cor[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10]-SIIfin_fit[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10])
std_s1 = np.std(data_cor[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]]-SIIfin_fit[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]])
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
print('		For SII2: '+str(std_s2/stadev)+' < 3')
print('		For SII1: '+str(std_s1/stadev)+' < 3')
# two components
std2_s2 = np.std(data_cor[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10]-SII2fin_fit[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10])
std2_s1 = np.std(data_cor[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]]-SII2fin_fit[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]])
print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
print('		For SII2: '+str(std2_s2/stadev)+' < 3')
print('		For SII1: '+str(std2_s1/stadev)+' < 3')

# We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
maxS1 = max(SIIfin_fit[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]]) #SIIfin_fit[np.where(abs(SIIresu.values['mu_0']-l)<0.28)[0][0]]
maxS2 = max(SIIfin_fit[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]]) #SIIfin_fit[np.where(abs(SIIresu.values['mu_1']-l)<0.28)[0][0]]
max2S1 = max(SII2fin_fit[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]]) #SII2fin_fit[np.where(abs(twoSIIresu.values['mu_0']-l)<0.28)[0][0]] 
max2S2 = max(SII2fin_fit[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]]) #SII2fin_fit[np.where(abs(twoSIIresu.values['mu_1']-l)<0.28)[0][0]] 
# Systemic velocity + error
vsys = v_luz*z
er_vsys = v_luz*erz

# one component
vS2 = v_luz*((SIIresu.values['mu_0']-l_SII_2)/l_SII_2)
sigS2 = pix_to_v*np.sqrt(SIIresu.values['sig_0']**2-sig_inst**2)

if SIIresu.params['mu_0'].stderr == None: 
     print('Problem determining the errors!')
     evS2,esigS2 = 0.,0.
elif SIIresu.params['mu_0'].stderr != None: 
     evS2 = ((v_luz/l_SII_2)*SIIresu.params['mu_0'].stderr)-er_vsys
     esigS2 = pix_to_v*np.sqrt(SIIresu.values['sig_0']*SIIresu.params['sig_0'].stderr)/(np.sqrt(SIIresu.values['sig_0']**2-sig_inst**2))

# two comps
v2S2 = v_luz*((twoSIIresu.values['mu_0']-l_SII_2)/l_SII_2)
v20S2 = v_luz*((twoSIIresu.values['mu_20']-l_SII_2)/l_SII_2)
sig2S2 = pix_to_v*np.sqrt(twoSIIresu.values['sig_0']**2-sig_inst**2)
sig20S2 = pix_to_v*np.sqrt(twoSIIresu.values['sig_20']**2-sig_inst**2)

if twoSIIresu.params['mu_20'].stderr == None:
    print('Problem determining the errors of the velocity 2!')
    ev20S2,ev2S2=0.,0.
elif twoSIIresu.params['mu_20'].stderr != None:
    ev2S2 = ((v_luz/l_SII_2)*twoSIIresu.params['mu_0'].stderr)-er_vsys
    ev20S2 = ((v_luz/l_SII_2)*twoSIIresu.params['mu_20'].stderr)-er_vsys

if twoSIIresu.params['sig_20'].stderr == None:
    print('Problem determining the errors of the sigma 2!')
    esig2S2,esig20S2 = 0.,0.
if twoSIIresu.params['sig_20'].stderr != None:
    esig2S2 = pix_to_v*np.sqrt(twoSIIresu.values['sig_0']*twoSIIresu.params['sig_0'].stderr)/(np.sqrt(twoSIIresu.values['sig_0']**2-sig_inst**2))
    esig20S2 = pix_to_v*np.sqrt(twoSIIresu.values['sig_20']*twoSIIresu.params['sig_20'].stderr)/(np.sqrt(twoSIIresu.values['sig_20']**2-sig_inst**2))

# Save the velocities for the 1 component fit
if os.path.exists(path+'eps_adjS1.txt'): os.remove(path+'eps_adjS1.txt')
np.savetxt(path+'v_sig_adjS1b.txt',np.c_[vS2,evS2,sigS2,esigS2],('%8.5f','%8.5f','%8.5f','%8.5f'),header=('v_SII2\tev_SII2\tsig_SII2\tesig_SII2'))

################################################### PLOT #########################################################
plt.close()
# MAIN plot
fig1   = plt.figure(1,figsize=(10, 9))
frame1 = fig1.add_axes((.1,.25,.85,.65)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor,'k')			     # Initial data
plt.plot(l,SIIfin_fit,'r-')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'y--',label='Linear fit')	#linestyle=(0, (5, 8)),
plt.plot(l,gaus1,'b-')
plt.plot(l,gaus2,'b-',label='Narrow component')
textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
		    r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(maxS2/maxS1)))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
if l11<6900.:
    x_frame = 6380.
else: 
    x_frame = 6800.
frame1.text(x_frame,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame2 = fig1.add_axes((.1,.1,.85,.15))
plt.plot(l,data_cor-SIIfin_fit,'k')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),c='grey',linestyle='--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+2*stadev,c='grey',linestyle='--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-2*stadev,c='grey',linestyle='--') 	# 3 sigma down limit
plt.ylim(-(3*stadev)*2,(3*stadev)*2)
plt.savefig(path+'adj_metS_SII_1comp.png')

###############################################################################################################
# Two components in SII
# MAIN plot
fig2   = plt.figure(2,figsize=(10, 9))
frame3 = fig2.add_axes((.1,.25,.85,.65)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
plt.plot(l,data_cor,'k')			     # Initial data
plt.plot(l,SII2fin_fit,'r-')
plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'y--',label='Linear fit')# linestyle=(0, (5, 8)),label='Linear fit')
plt.plot(l,gaus21,'b-')
plt.plot(l,gaus22,'b-',label='Narrow component')
plt.plot(l,gaus23,c='darkmagenta',linestyle='-')
plt.plot(l,gaus24,c='darkmagenta',linestyle='-',label='Secondary component')
textstr = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2S2,ev2S2),
		    r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v20S2,ev20S2),
		    r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2S2,esig2S2),
		    r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig20S2,esig20S2),
		    r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(max2S2/max2S1)))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
frame3.text(x_frame,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated

frame3.set_xticklabels([]) 			# Remove x-tic labels for the first frame
plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.xlim(l[0],l[-1])
plt.legend(loc='best')

# RESIDUAL plot
frame4 = fig2.add_axes((.1,.1,.85,.15))
plt.plot(l,data_cor-SII2fin_fit,'k')		# Main
plt.xlabel('Wavelength ($\AA$)',fontsize=16)
plt.ylabel('Residuals',fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.xlim(l[0],l[-1])
plt.plot(l,np.zeros(len(l)),color='grey',linestyle='--')         	# Line around zero
plt.plot(l,np.zeros(len(l))+2*stadev,color='grey',linestyle='--')	# 3 sigma upper limit
plt.plot(l,np.zeros(len(l))-2*stadev,color='grey',linestyle='--') 	# 3 sigma down limit
plt.ylim(-(3*stadev)*2,(3*stadev)*2)
plt.savefig(path+'adj_metS_SII_2comp.png')

##########################################################################################################################################################
# We make an F-test to see if it is significant the presence of a second component in the lines. 
# As the only possible method here is the S-method due to the fact that there are no O-lines in this spectra, 
# then the method can only be applied to the SII lines (so the wavelength range would be around this two lines)
pre_x = data_cor[np.where(l_init>l3)[0][0]-20:np.where(l_init<l2)[0][-1]+20]-SIIfin_fit[np.where(l_init>l3)[0][0]-20:np.where(l_init<l2)[0][-1]+20]
pre_y = data_cor[np.where(l_init>l3)[0][0]-20:np.where(l_init<l2)[0][-1]+20]-SII2fin_fit[np.where(l_init>l3)[0][0]-20:np.where(l_init<l2)[0][-1]+20]
tx, ty = stats.obrientransform(pre_x, pre_y)
fvalue1, pvalue1 = stats.f_oneway(tx,ty)
fvalue, pvalue = stats.f_oneway(pre_x,pre_y)
fvalue2, pvalue2 = stats.levene(pre_x,pre_y)
##########################################################################################################################################################
print('')
if SIIresu.chisqr < twoSIIresu.chisqr: 
    print('The probability of a second component (one component vs two components) in this spectra with the F-test of IDL is: NOT CALCULATED. ONLY ONE COMPONENT AVAILABLE')
    print('Chi-square of the two component-fit bigger than the chi-square of the one component fit')
    print('Do not trust in the next statistics!')
    print('')
#else: 
#    fstat = ftest(SIIresu.chisqr,twoSIIresu.chisqr,SIIresu.nfree,twoSIIresu.nfree)
#    print('The probability of a second component (one component vs two components) in this spectra with the F-test of IDL is: '+str(fstat['p-value']))
print('The probability of a second component (one component vs two components) in this spectra with the F-test is: '+str(pvalue))
print('The probability of a second component (one component vs two components) in this spectra with the F-test (and O Brien) is: '+str(pvalue1))
print('The probability of a second component (one component vs two components) in this spectra with the Levene-test is: '+str(pvalue2))
print('')

##########################################################################################################################################################
#---------------------------------------------------------------------------------------------------------------------------------------------------------
##########################################################################################################################################################

# Select if one or two components in the SII lines and then apply to the rest
if SIIresu.chisqr < twoSIIresu.chisqr: 
    trigger = 'Y'
else: 
    trigger = input('Is the fit good enough with one component? ("Y"/"N"): ')

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
    cd = lmfit.Parameter('mu_0', value = SIIresu.values["mu_0"],vary=False)
    de = lmfit.Parameter('sig_0', value = SIIresu.values["sig_0"],vary=False)
    ef = lmfit.Parameter('amp_0', value = SIIresu.values["amp_0"],vary=False)
    fg = lmfit.Parameter('mu_1', value = SIIresu.values["mu_1"],vary=False)
    gh = lmfit.Parameter('sig_1', value = SIIresu.values["sig_1"],vary=False)
    hi = lmfit.Parameter('amp_1', value = SIIresu.values["amp_1"],vary=False)
    ij = lmfit.Parameter('mu_2', value = mu2,expr='mu_0*(6583.45/6730.82)')
    jk = lmfit.Parameter('sig_2', value = sig2,expr='sig_0')
    kl = lmfit.Parameter('amp_2', value = amp2,min=0.)
    lm = lmfit.Parameter('mu_3', value = mu3,expr='mu_0*(6562.801/6730.82)')
    mn = lmfit.Parameter('sig_3', value = sig3,expr='sig_0')
    no = lmfit.Parameter('amp_3', value = amp3,min=0.)
    op = lmfit.Parameter('mu_4', value = mu4,expr='mu_0*(6548.05/6730.82)')
    pq = lmfit.Parameter('sig_4', value = sig4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value = amp4,min=0.,expr='amp_2*(1./3.)')
    params.add_many(sl,it,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr)

    resu1 = comp_mod.fit(data_cor,params,x=l)
    print('El resultado del fit: '+resu1.message)
    lmfit.model.save_modelresult(resu1, path+'one_modelresult.sav')
    with open(path+'fitone_result.txt', 'w') as fh:
         fh.write(resu1.fit_report())

    ######################################## Calculate gaussians and final fit ################################################
    # Now we create and plot the individual gaussians of the fit
    gaus1 = noOfuncts.gaussian(l,resu1.values['mu_0'],resu1.values['sig_0'],resu1.values['amp_0']) 
    gaus2 = noOfuncts.gaussian(l,resu1.values['mu_1'],resu1.values['sig_1'],resu1.values['amp_1'])
    gaus3 = noOfuncts.gaussian(l,resu1.values['mu_2'],resu1.values['sig_2'],resu1.values['amp_2']) 
    gaus4 = noOfuncts.gaussian(l,resu1.values['mu_3'],resu1.values['sig_3'],resu1.values['amp_3'])
    gaus5 = noOfuncts.gaussian(l,resu1.values['mu_4'],resu1.values['sig_4'],resu1.values['amp_4'])
    fin_fit = resu1.best_fit

    # one component
    stdf_s2 = np.std(data_cor[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10]-fin_fit[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10])
    stdf_s1 = np.std(data_cor[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]]-fin_fit[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]])
    stdf_n2 = np.std(data_cor[np.where(l_init<l5)[0][-1]:np.where(l_init>l6)[0][0]+10]-fin_fit[np.where(l_init<l5)[0][-1]:np.where(l_init>l6)[0][0]+10])
    stdf_ha = np.std(data_cor[np.where(l_init<l7)[0][-1]:np.where(l_init>l8)[0][0]]-fin_fit[np.where(l_init<l7)[0][-1]:np.where(l_init>l8)[0][0]])
    stdf_n1 = np.std(data_cor[np.where(l_init<l9)[0][-1]-10:np.where(l_init>l10)[0][0]]-fin_fit[np.where(l_init<l9)[0][-1]-10:np.where(l_init>l10)[0][0]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
    print('	For SII2: '+str(stdf_s2/stadev)+' < 3')
    print('	For SII1: '+str(stdf_s1/stadev)+' < 3')
    print('	For NII2: '+str(stdf_n2/stadev)+' < 3')
    print('	For Halpha: '+str(stdf_ha/stadev)+' < 3')
    print('	For NII1: '+str(stdf_n1/stadev)+' < 3')
    
    if os.path.exists(path+'eps_adjS1.txt'): os.remove(path+'eps_adjS1.txt')
    np.savetxt(path+'eps_adjS1.txt',np.c_[stdf_s2/stadev,stdf_s1/stadev,stdf_n2/stadev,stdf_ha/stadev,stdf_n1/stadev,resu1.chisqr],('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('SII2\tSII1\tNII2\tHa\tNII1\tChi2'))

    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    maxfS1 = fin_fit[np.where(abs(resu1.values['mu_0']-l)<0.5)[0][0]] #max(fin_fit[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]])
    maxfS2 = fin_fit[np.where(abs(resu1.values['mu_1']-l)<0.5)[0][0]] #max(fin_fit[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]])
    maxfN1 = fin_fit[np.where(abs(resu1.values['mu_2']-l)<0.5)[0][0]] #max(fin_fit[np.where(l_init>l9)[0][0]:np.where(l_init<l10)[0][-1]])
    maxfHa = fin_fit[np.where(abs(resu1.values['mu_3']-l)<0.5)[0][0]] #max(fin_fit[np.where(l_init>l7)[0][0]:np.where(l_init<l8)[0][-1]])
    maxfN2 = fin_fit[np.where(abs(resu1.values['mu_4']-l)<0.5)[0][0]] #max(fin_fit[np.where(l_init>l5)[0][0]:np.where(l_init<l6)[0][-1]])

    ###################################################### PLOT ##############################################################
    plt.close('all')
    # MAIN plot
    fig1   = plt.figure(1,figsize=(10, 9))
    frame1 = fig1.add_axes((.1,.25,.85,.65)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(l,data_cor,'k')			     # Initial data
    plt.plot(l,fin_fit,'r-')
    plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),'y--',label='Linear fit')	# linestyle=(0, (5, 8)),
    plt.plot(l,gaus1,'b-')
    plt.plot(l,gaus2,'b-')
    plt.plot(l,gaus3,'b-')
    plt.plot(l,gaus4,'b-')
    plt.plot(l,gaus5,'b-',label='Narrow component')
    textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
		    r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(maxfS2/maxfS1),
		    r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxfHa)+' $10^{-14}$',
		    r'$\frac{F_{NII_{2}}}{F_{NII_{1}}}$ = '+ '{:.3f}'.format(maxfN2/maxfN1)))
    plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
    frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
    plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=17)
    plt.tick_params(axis='both', labelsize=15)
    plt.xlim(l[0],l[-1])
    if l[0]<l_OI_1:                           
        plt.axvline(l_OI_1,0,maxfHa*2+0.1,color='peachpuff')
        plt.axvline(l_OI_2,0,maxfHa*2+0.1,color='peachpuff')
    elif l[0]<l_OI_2 and l[0]>l_OI_1: plt.axvline(l_OI_2,0,maxfHa*2+1,color='peachpuff')
    plt.legend(loc='best',fontsize='large')

    # RESIDUAL plot
    frame2 = fig1.add_axes((.1,.1,.85,.15))
    plt.plot(l,data_cor-fin_fit,'k')		# Main
    plt.xlabel('Wavelength ($\AA$)',fontsize=17)
    plt.ylabel('Residuals',fontsize=17)
    plt.tick_params(axis='both', labelsize=15)
    plt.xlim(l[0],l[-1])
    plt.plot(l,np.zeros(len(l)),color='grey',linestyle='--')         	# Line around zero
    plt.plot(l,np.zeros(len(l))+2*stadev,color='grey',linestyle='--')	# 3 sigma upper limit
    plt.plot(l,np.zeros(len(l))-2*stadev,color='grey',linestyle='--') 	# 3 sigma down limit
    if l[0]<l_OI_1: 
        plt.axvline(l_OI_1,-3,3,color='peachpuff')#,linestyle='--')
        plt.axvline(l_OI_2,-3,3,color='peachpuff')#,linestyle='--')
    elif l[0]<l_OI_2 and l[0]>l_OI_1: plt.axvline(l_OI_2,-3,3,color='peachpuff')
    plt.ylim(-(3*stadev)*3,(3*stadev)*3)

    plt.savefig(path+'adj_metS_SII_full_1comp.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    frame1.text(x_frame,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
    plt.savefig(path+'adj_metS_SII_full_1comp.png',bbox_inches='tight',pad_inches=0.2)
    
    #########################################################################################################################    
    trigger2 = input('Do the fit needs a broad Halpha component? ("Y"/"N"): ')
    if trigger2 == 'N': 
        np.savetxt(path+'fitone_best_values.txt',np.c_[resu1.data,resu1.best_fit,lin_data_fin],fmt=('%5.10f','%5.10f','%5.10f'),header=('Real_data\tBest_fit\tLineal_fit'))	
        print('The final plots are already printed and have been saved!')
    elif trigger2 == 'Y':
	# Now we define the initial guesses and the constraints
        newxb = l[np.where(l_init>l9)[0][0]:np.where(l_init<l6)[0][-1]]
        newyb = data_cor[np.where(l_init>l9)[0][0]:np.where(l_init<l6)[0][-1]]
        sigb = 16.
        mub  = mu3
        ampb = amp3/3.
        paramsbH = lmfit.Parameters()
        # broad components
        ab = lmfit.Parameter('mu_b',value=mub)
        bc = lmfit.Parameter('sig_b',value=sigb,min=sig_inst)
        rs = lmfit.Parameter('amp_b',value=ampb)#,min=resu1.values['amp_3']/8.,max=resu1.values['amp_3'])
        paramsbH.add_many(sl,it,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,ab,bc,rs)

        broadresu = broad_mod.fit(data_cor,paramsbH,x=l)
        print('El resultado del fit: '+broadresu.message)
        lmfit.model.save_modelresult(broadresu,path+'broadone_modelresult.sav')
        with open(path+'fitbroad_result.txt', 'w') as fh:
            fh.write(broadresu.fit_report())

        ##################################### Calculate gaussians and final fit ############################################
	# Now we create and plot the individual gaussians of the fit
        bgaus1 = noOfuncts.gaussian(l,broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0']) 
        bgaus2 = noOfuncts.gaussian(l,broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'])
        bgaus3 = noOfuncts.gaussian(l,broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2'])
        bgaus4 = noOfuncts.gaussian(l,broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'])
        bgaus5 = noOfuncts.gaussian(l,broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'])
        bgaus6 = noOfuncts.gaussian(l,broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])
        broad_fit = broadresu.best_fit

        # We have to calculate the contribution of each component to the global fit
        # Lets define the linear fit data to add to each individual gaussian
        bgaus_total = broad_fit - lin_data_fin
        np.savetxt(path+'fitbroad_best_values.txt',np.c_[broadresu.data,broadresu.best_fit,lin_data_fin,bgaus3,bgaus4,bgaus5,bgaus6],fmt=('%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f'),header=('Real_data\tBest_fit\tLineal_fit\tNarrow_NII2\tNarrow_Halpha\tNarrow_NII1\tBroad_Halpha'))
        # Now lets determine the contribution of the individual components as follows:
        contr_HaN = sum(bgaus4)
        contr_HaB = sum(bgaus6)
        ix_Br_sup = np.where(bgaus6 > 10**-5)[0][-1]
        ix_Br_inf = np.where(bgaus6 > 10**-5)[0][0]
        contr_NII2N = sum(bgaus5)
        contr_NII1N = sum(bgaus3)
        total_flux_NII_Halp = sum(bgaus_total[ix_Br_inf:ix_Br_sup])

        contr_HaBtoNHa = 100*(contr_HaB/total_flux_NII_Halp)
        contr_HaNtoNHa = 100*(contr_HaN/total_flux_NII_Halp)
        contr_NII2NtoNHa = 100*(contr_NII2N/total_flux_NII_Halp)
        contr_NII1NtoNHa = 100*(contr_NII1N/total_flux_NII_Halp)

        print('The contribution of the broad component to the total Halpha+N flux is: '+'{:.2f}'.format(contr_HaBtoNHa)+'%')
        np.savetxt(path+'1cBroad_N+Ha_indivcontr.txt',np.c_[contr_NII1NtoNHa,contr_HaNtoNHa,contr_HaBtoNHa,contr_NII2NtoNHa],fmt=('%10.7f','%10.7f','%10.7f','%10.7f'),header=('Narrow_NII1(%)\tNarrow_Halpha(%)\tBroad_Halpha(%)\tNarrow_NII2(%)'))

        # Now we calculate the epsilon values derived from the fit
        stdb_s2 = np.std(data_cor[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10]-broad_fit[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10])
        stdb_s1 = np.std(data_cor[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]]-broad_fit[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]])
        stdb_n2 = np.std(data_cor[np.where(l_init<l5)[0][-1]:np.where(l_init>l6)[0][0]+10]-broad_fit[np.where(l_init<l5)[0][-1]:np.where(l_init>l6)[0][0]+10])
        stdb_ha = np.std(data_cor[np.where(l_init<l7)[0][-1]:np.where(l_init>l8)[0][0]]-broad_fit[np.where(l_init<l7)[0][-1]:np.where(l_init>l8)[0][0]])
        stdb_n1 = np.std(data_cor[np.where(l_init<l9)[0][-1]-10:np.where(l_init>l10)[0][0]]-broad_fit[np.where(l_init<l9)[0][-1]-10:np.where(l_init>l10)[0][0]])
        print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component + Ha is... ')
        print('		For SII2: '+str(stdb_s2/stadev)+' < 3')
        print('		For SII1: '+str(stdb_s1/stadev)+' < 3')
        print('		For NII2: '+str(stdb_n2/stadev)+' < 3')
        print('		For Halp: '+str(stdb_ha/stadev)+' < 3')
        print('		For SII1: '+str(stdb_n1/stadev)+' < 3')
    	
        if os.path.exists(path+'eps_adjS1b.txt'): os.remove(path+'eps_adjS1b.txt')
        np.savetxt(path+'eps_adjS1b.txt',np.c_[stdb_s2/stadev,stdb_s1/stadev,stdb_n2/stadev,stdb_ha/stadev,stdb_n1/stadev,broadresu.chisqr],
                        ('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('SII2\tSII1\tNII2\tHa\tNII1\tChi2'))

   	# We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
        maxbS1 = broad_fit[np.where(abs(broadresu.values['mu_0']-l)<0.5)[0][0]] #max(broad_fit[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]])
        maxbS2 = broad_fit[np.where(abs(broadresu.values['mu_1']-l)<0.5)[0][0]] #max(broad_fit[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]])
        maxbN1 = broad_fit[np.where(abs(broadresu.values['mu_2']-l)<0.5)[0][0]] #max(broad_fit[np.where(l_init>l9)[0][0]:np.where(l_init<l10)[0][-1]])
        maxbHa = broad_fit[np.where(abs(broadresu.values['mu_3']-l)<0.5)[0][0]] #max(broad_fit[np.where(l_init>l7)[0][0]:np.where(l_init<l8)[0][-1]])
        maxbN2 = broad_fit[np.where(abs(broadresu.values['mu_4']-l)<0.5)[0][0]] #max(broad_fit[np.where(l_init>l5)[0][0]:np.where(l_init<l6)[0][-1]])
        # two comps
        vbS2 = v_luz*((broadresu.values['mu_0']-l_SII_2)/l_SII_2)
        vb0S2 = v_luz*((broadresu.values['mu_b']-l_Halpha)/l_Halpha)
        sigbS2 = pix_to_v*np.sqrt(broadresu.values['sig_0']**2-sig_inst**2)
        sigb0S2 = pix_to_v*np.sqrt(broadresu.values['sig_b']**2-sig_inst**2)
        
        if SIIresu.params['mu_0'].stderr == None: 
            print('Problem determining the errors! First component ')
            evbS2 = 0.
        elif SIIresu.params['mu_0'].stderr != None: 
            evbS2 = ((v_luz/l_SII_2)*SIIresu.params['mu_0'].stderr)-er_vsys
        if SIIresu.params['sig_0'].stderr == None:
            print('Problem determining the errors! First component sigma')
            esigbS2 = 0.
        elif SIIresu.params['sig_0'].stderr != None: 
            esigbS2 = pix_to_v*np.sqrt(broadresu.values['sig_0']*SIIresu.params['sig_0'].stderr)/(np.sqrt(broadresu.values['sig_0']**2-sig_inst**2))

        if broadresu.params['mu_b'].stderr == None:
            print('Problem determining the errors! Broad component ')
            evb0S2 = 0.
        elif broadresu.params['mu_b'].stderr != None:
            evb0S2 = ((v_luz/l_SII_2)*broadresu.params['mu_b'].stderr)-er_vsys
        if broadresu.params['sig_b'].stderr == None:
            print('Problem determining the errors! Broad component sigma')
            esigb0S2 = 0.
        elif broadresu.params['sig_b'].stderr != None:
            esigb0S2 = pix_to_v*np.sqrt(broadresu.values['sig_b']*broadresu.params['sig_b'].stderr)/(np.sqrt(broadresu.values['sig_b']**2-sig_inst**2))

        if os.path.exists(path+'v_sig_adjS1b.txt'): os.remove(path+'v_sig_adjS1b.txt')
        np.savetxt(path+'v_sig_adjS1b.txt',np.c_[vbS2,evbS2,vb0S2,evb0S2,sigbS2,esigbS2,sigb0S2,esigb0S2],
                ('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('v_SII2\tev_SII2\tv_bSII2\tev_bSII2\tsig_SII2\tesig_SII2\tsig_bSII2\tesig_bSII2'))
        '''
	################################################################################################################
	# We make an F-test to see if it is significant the presence of a broad component in the lines. 
        pre_x = data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-fin_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]
        pre_y = data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-broad_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]
        tx, ty = stats.obrientransform(pre_x, pre_y)
        fvalue1, pvalue1 = stats.f_oneway(tx,ty)
        fbvalue, pbvalue = stats.f_oneway(data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-fin_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20],
					  data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-broad_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20])
        statsvalue, pbvalue2 = stats.levene(data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-fin_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20],
					    data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-broad_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20])
#        fstat = ftest(resu1.chisqr,broadresu.chisqr,resu1.nfree,broadresu.nfree)
        print('')
#        print('The probability of a second component (one component vs one + broad Halpha components) in this spectra with the F-test of IDL is: '+str(fstat['p-value']))
        print('The probability of a second component (one component vs one + broad Halpha components) in this spectra with the F-test is: '+str(pbvalue))
        print('The probability of a second component (one component vs one + broad Halpha components) in this spectra with the F-test (and O Brien) is: '+str(pvalue1))
        print('The probability of a second component (one component vs one + broad Halpha components) in this spectra with the Levene-test is: '+str(pbvalue2))
        print('')
        ''' 
   	################################################## PLOT ######################################################
        plt.close('all')
    	# MAIN plot
        fig1   = plt.figure(1,figsize=(10, 9))
        frame1 = fig1.add_axes((.1,.25,.85,.65)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
        plt.plot(l,data_cor,'k')			     # Initial data
        plt.plot(l,broad_fit,'r-')
        plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),c='y',linestyle=(0, (5, 8)),label='Linear fit')
        plt.plot(l,bgaus1,'b-')
        plt.plot(l,bgaus2,'b-')
        plt.plot(l,bgaus3,'b-')
        plt.plot(l,bgaus4,'b-')
        plt.plot(l,bgaus5,'b-',label='Narrow component')
        plt.plot(l,bgaus6,c='darkorange',linestyle='-',label='Broad component')
        textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vbS2,evbS2),
			r'$V_{SII_{2-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(vb0S2,evb0S2),
		    	r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigbS2,esigbS2),
		    	r'$\sigma_{SII_{2-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(sigb0S2,esigb0S2),
			r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(maxbS2/maxbS1),
		    	r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxbHa)+' $10^{-14}$',
		        r'$\frac{F_{NII_{2}}}{F_{NII_{1}}}$ = '+ '{:.3f}'.format(maxbN2/maxbN1)))
        plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
        frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
        plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=17)
        plt.tick_params(axis='both', labelsize=15)
        plt.xlim(l[0],l[-1])
        plt.legend(loc='best',fontsize='large')
        if l[0]<l_OI_1:
            plt.axvline(l_OI_1,0,maxbHa*2+0.1,color='peachpuff')#,linestyle='--')
            plt.axvline(l_OI_2,0,maxbHa*2+0.1,color='peachpuff')
        elif l[0]<l_OI_2 and l[0]>l_OI_1: plt.axvline(l_OI_2,0,maxbHa*2+1,color='peachpuff')
	# RESIDUAL plot
        frame2 = fig1.add_axes((.1,.1,.85,.15))
        plt.plot(l,data_cor-broad_fit,'k')		# Main
        plt.xlabel('Wavelength ($\AA$)',fontsize=17)
        plt.ylabel('Residuals',fontsize=17)
        plt.tick_params(axis='both', labelsize=15)
        plt.xlim(l[0],l[-1])
        plt.plot(l,np.zeros(len(l)),color='grey',linestyle='--')         	# Line around zero
        plt.plot(l,np.zeros(len(l))+2*stadev,color='grey',linestyle='--')	# 3 sigma upper limit
        plt.plot(l,np.zeros(len(l))-2*stadev,color='grey',linestyle='--') 	# 3 sigma down limit
        if l[0]<l_OI_1:
            plt.axvline(l_OI_1,-3,3,color='peachpuff')#,linestyle='--')
            plt.axvline(l_OI_2,-3,3,color='peachpuff')
        elif l[0]<l_OI_2 and l[0]>l_OI_1: plt.axvline(l_OI_2,-3,3,color='peachpuff')
        plt.ylim(-(3*stadev)*3,(3*stadev)*3)

        plt.savefig(path+'adj_metS_SII_full_1comp_broadH.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        frame1.text(x_frame,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
        plt.savefig(path+'adj_metS_SII_full_1comp_broadH.png',bbox_inches='tight',pad_inches=0.2)

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
    no = lmfit.Parameter('amp_3', value=amp3,min=0.1)
    op = lmfit.Parameter('mu_4', value=mu4,expr='mu_0*(6548./6731.)')
    pq = lmfit.Parameter('sig_4', value=sig4,expr='sig_0')
    qr = lmfit.Parameter('amp_4', value=amp4,expr='amp_2*(1./3.)',vary=False)
    # second components
    aaa = lmfit.Parameter('mu_20', value=twoSIIresu.values["mu_20"],vary=False) #-1.0
    aab = lmfit.Parameter('sig_20', value=twoSIIresu.values["sig_20"],vary=False)
    aac = lmfit.Parameter('amp_20', value=twoSIIresu.values["amp_20"],vary=False)
    aad = lmfit.Parameter('mu_21', value=twoSIIresu.values["mu_21"],vary=False)
    aae = lmfit.Parameter('sig_21', value=twoSIIresu.values["sig_21"],vary=False)
    aaf = lmfit.Parameter('amp_21', value=twoSIIresu.values["amp_21"],vary=False)
    aag = lmfit.Parameter('mu_22', value=mu2,expr='mu_20*(6583.45/6730.82)')
    aah = lmfit.Parameter('sig_22', value=sig22,expr='sig_20')
    aai = lmfit.Parameter('amp_22', value=amp22,min=0.)
    aaj = lmfit.Parameter('mu_23', value=mu3,expr='mu_20*(6563.45/6730.82)')
    aak = lmfit.Parameter('sig_23', value=sig23,expr='sig_20')
    aal = lmfit.Parameter('amp_23', value=amp23,min=0.1)
    aam = lmfit.Parameter('mu_24', value=mu4,expr='mu_20*(6548.05/6730.82)')
    aan = lmfit.Parameter('sig_24', value=sig24,expr='sig_20')
    aao = lmfit.Parameter('amp_24', value=amp24,expr='amp_22*(1./3.)',vary=False)
    params2c.add_many(sl,it,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,aaa,aab,aac,aad,aae,aaf,aag,aah,aai,aaj,aak,aal,aam,aan,aao)

    twocompresu = twocomp_mod.fit(data_cor,params2c,x=l)
    print('El resultado del fit: '+twocompresu.message)
    lmfit.model.save_modelresult(twocompresu, path+'two_modelresult.sav')
    with open(path+'fittwocomp_result.txt', 'w') as fh:
             fh.write(twocompresu.fit_report())
    ##################################### Calculate gaussians and final fit ##########################################
    # Now we create and plot the individual gaussians of the fit
    tgaus1 = noOfuncts.gaussian(l,twocompresu.values['mu_0'],twocompresu.values['sig_0'],twocompresu.values['amp_0'])
    tgaus2 = noOfuncts.gaussian(l,twocompresu.values['mu_1'],twocompresu.values['sig_1'],twocompresu.values['amp_1'])
    tgaus3 = noOfuncts.gaussian(l,twocompresu.values['mu_2'],twocompresu.values['sig_2'],twocompresu.values['amp_2'])
    tgaus4 = noOfuncts.gaussian(l,twocompresu.values['mu_3'],twocompresu.values['sig_3'],twocompresu.values['amp_3'])
    tgaus5 = noOfuncts.gaussian(l,twocompresu.values['mu_4'],twocompresu.values['sig_4'],twocompresu.values['amp_4'])
    tgaus6 = noOfuncts.gaussian(l,twocompresu.values['mu_20'],twocompresu.values['sig_20'],twocompresu.values['amp_20'])
    tgaus7 = noOfuncts.gaussian(l,twocompresu.values['mu_21'],twocompresu.values['sig_21'],twocompresu.values['amp_21']) 
    tgaus8 = noOfuncts.gaussian(l,twocompresu.values['mu_22'],twocompresu.values['sig_22'],twocompresu.values['amp_22'])
    tgaus9 = noOfuncts.gaussian(l,twocompresu.values['mu_23'],twocompresu.values['sig_23'],twocompresu.values['amp_23'])
    tgaus10 = noOfuncts.gaussian(l,twocompresu.values['mu_24'],twocompresu.values['sig_24'],twocompresu.values['amp_24'])
    fin2_fit = twocompresu.best_fit

    # two components
    stdf2_s2 = np.std(data_cor[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10]-fin2_fit[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10])
    stdf2_s1 = np.std(data_cor[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]]-fin2_fit[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]])
    stdf2_n2 = np.std(data_cor[np.where(l_init<l5)[0][-1]:np.where(l_init>l6)[0][0]+10]-fin2_fit[np.where(l_init<l5)[0][-1]:np.where(l_init>l6)[0][0]+10])
    stdf2_ha = np.std(data_cor[np.where(l_init<l7)[0][-1]:np.where(l_init>l8)[0][0]]-fin2_fit[np.where(l_init<l7)[0][-1]:np.where(l_init>l8)[0][0]])
    stdf2_n1 = np.std(data_cor[np.where(l_init<l9)[0][-1]-10:np.where(l_init>l10)[0][0]]-fin2_fit[np.where(l_init<l9)[0][-1]-10:np.where(l_init>l10)[0][0]])
    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
    print('	For SII2: '+str(stdf2_s2/stadev)+' < 3')
    print('	For SII1: '+str(stdf2_s1/stadev)+' < 3')
    print('	For NII2: '+str(stdf2_n2/stadev)+' < 3')
    print('	For Halp: '+str(stdf2_ha/stadev)+' < 3')
    print('	For NII1: '+str(stdf2_n1/stadev)+' < 3')
    
    if os.path.exists(path+'eps_adjS2.txt'): os.remove(path+'eps_adjS2.txt')
    np.savetxt(path+'eps_adjS2.txt',np.c_[stdf2_s2/stadev,stdf2_s1/stadev,stdf2_n2/stadev,stdf2_ha/stadev,stdf2_n1/stadev,twocompresu.chisqr],
    		('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('SII2\tSII1\tNII2\tHa\tNII1\tChi2'))

    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    maxb2S1 = fin2_fit[np.where(abs(twocompresu.values['mu_0']-l)<0.5)[0][0]] #max(fin2_fit[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]])
    maxb2S2 = fin2_fit[np.where(abs(twocompresu.values['mu_1']-l)<0.5)[0][0]] #max(fin2_fit[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]])
    maxb2N1 = fin2_fit[np.where(abs(twocompresu.values['mu_2']-l)<0.52)[0][0]] #max(fin2_fit[np.where(l_init>l9)[0][0]:np.where(l_init<l10)[0][-1]])
    maxb2Ha = fin2_fit[np.where(abs(twocompresu.values['mu_3']-l)<0.5)[0][0]] #max(fin2_fit[np.where(l_init>l7)[0][0]:np.where(l_init<l8)[0][-1]])
    maxb2N2 = fin2_fit[np.where(abs(twocompresu.values['mu_4']-l)<0.5)[0][0]] #max(fin2_fit[np.where(l_init>l5)[0][0]:np.where(l_init<l6)[0][-1]])
    # two comps
    vb2S2 = v_luz*((twocompresu.values['mu_0']-l_SII_2)/l_SII_2)
    vb20S2 = v_luz*((twocompresu.values['mu_20']-l_SII_2)/l_SII_2)
    evb2S2 = ((v_luz/l_SII_2)*twoSIIresu.params['mu_0'].stderr)-er_vsys
    evb20S2 = ((v_luz/l_SII_2)*twoSIIresu.params['mu_20'].stderr)-er_vsys
    sigb2S2 = pix_to_v*np.sqrt(twocompresu.values['sig_0']**2-sig_inst**2)
    sigb20S2 = pix_to_v*np.sqrt(twocompresu.values['sig_20']**2-sig_inst**2)
    esigb2S2 = pix_to_v*np.sqrt(twocompresu.values['sig_0']*twoSIIresu.params['sig_0'].stderr)/(np.sqrt(twocompresu.values['sig_0']**2-sig_inst**2))
    esigb20S2 = pix_to_v*np.sqrt(twocompresu.values['sig_20']*twoSIIresu.params['sig_20'].stderr)/(np.sqrt(twocompresu.values['sig_20']**2-sig_inst**2))

    if os.path.exists(path+'v_sig_adjS2.txt'): os.remove(path+'v_sig_adjS2.txt')
    np.savetxt(path+'v_sig_adjS2.txt',np.c_[vb2S2,evb2S2,vb20S2,evb20S2,sigb2S2,esigb2S2,sigb20S2,esigb20S2],
    		  ('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('v_SII2\tev_SII2\tv_2SII2\tev_2SII2\tsig_SII2\tesig_SII2\tsig_2SII2\tesig_2SII2'))

    ################################################# PLOT ########################################################
    plt.close('all')
    # MAIN plot
    fig1   = plt.figure(1,figsize=(10, 9))
    frame1 = fig1.add_axes((.1,.25,.85,.65)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(l,data_cor,'k')			     # Initial data
    plt.plot(l,fin2_fit,'r-')
    plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),c='y',linestyle=(0, (5, 8)),label='Linear fit')
    plt.plot(l,tgaus1,'b-')
    plt.plot(l,tgaus2,'b-')
    plt.plot(l,tgaus3,'b-')
    plt.plot(l,tgaus4,'b-')
    plt.plot(l,tgaus5,'b-',label='Narrow component')
    plt.plot(l,tgaus6,c='darkviolet',linestyle='-')
    plt.plot(l,tgaus7,c='darkviolet',linestyle='-')
    plt.plot(l,tgaus8,c='darkviolet',linestyle='-')
    plt.plot(l,tgaus9,c='darkviolet',linestyle='-')
    plt.plot(l,tgaus10,c='darkviolet',linestyle='-',label='Secondary component')
    textstr = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vb2S2,evb2S2),
		    r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vb20S2,evb20S2),
		    r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigb2S2,esigb2S2),
		    r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigb20S2,esigb20S2),
		    r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(maxb2S2/maxb2S1),
		    r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxb2Ha)+' $10^{-14}$',
		    r'$\frac{F_{NII_{2}}}{F_{NII_{1}}}$ = '+ '{:.3f}'.format(maxb2N2/maxb2N1)))
    plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
    frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
    plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=17)
    plt.tick_params(axis='both', labelsize=15)
    plt.xlim(l[0],l[-1])
    if l[0]<l_OI_1:
        plt.axvline(l_OI_1,0,maxb2Ha*2+0.1,color='peachpuff')
        plt.axvline(l_OI_2,0,maxb2Ha*2+0.1,color='peachpuff')
    elif l[0]<l_OI_2 and l[0]>l_OI_1: plt.axvline(l_OI_2,0,maxb2Ha*2+1,color='peachpuff')    
    plt.legend(loc='best')

    # RESIDUAL plot
    frame2 = fig1.add_axes((.1,.1,.85,.15))
    plt.plot(l,data_cor-fin2_fit,color='k')		# Main
    plt.xlabel('Wavelength ($\AA$)',fontsize=17)
    plt.ylabel('Residuals',fontsize=17)
    plt.tick_params(axis='both', labelsize=15)
    plt.xlim(l[0],l[-1])
    plt.plot(l,np.zeros(len(l)),color='grey',linestyle='--')         	# Line around zero
    plt.plot(l,np.zeros(len(l))+2.*stadev,color='grey',linestyle='--')	# 3 sigma upper limit
    plt.plot(l,np.zeros(len(l))-2.*stadev,color='grey',linestyle='--') 	# 3 sigma down limit
    if l[0]<l_OI_1:
        plt.axvline(l_OI_1,-3,3,color='peachpuff')#,linestyle='--')
        plt.axvline(l_OI_2,-3,3,color='peachpuff')
    elif l[0]<l_OI_2 and l[0]>l_OI_1: plt.axvline(l_OI_2,-3,3,color='peachpuff')
    plt.ylim(-(3*stadev)*3,(3*stadev)*3)

    plt.savefig(path+'adj_metS_SII_full_2comp.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    frame1.text(x_frame,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
    plt.savefig(path+'adj_metS_SII_full_2comp.png',bbox_inches='tight',pad_inches=0.2)

    #########################################################################################################################    
    trigger2 = input('Do the fit needs a broad Halpha component? ("Y"/"N"): ')
    if trigger2 == 'N': 
        print('The final plots are already printed and have been saved!')
        np.savetxt(path+'fittwo_best_values.txt',np.c_[twocompresu.data,twocompresu.best_fit,lin_data_fin],fmt=('%5.10f','%5.10f','%5.10f'),header=('Real_data\tBest_fit\tLineal_fit'))
    elif trigger2 == 'Y':
        # Now we define the initial guesses and the constraints
        sigb = 16.
        mub  = mu3
        ampb = amp3
        paramsbH = lmfit.Parameters()
	# broad components
        ab = lmfit.Parameter('mu_b',value=mub)
        bc = lmfit.Parameter('sig_b',value=sigb,min=sig_inst)
        rs = lmfit.Parameter('amp_b',value=ampb,min=0.)#,min=twocompresu.values['amp_3']/8.,max=twocompresu.values['amp_23'])
        paramsbH.add_many(sl,it,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,aaa,aab,aac,aad,aae,aaf,aag,aah,aai,aaj,aak,aal,aam,aan,aao,ab,bc,rs)

        twobroadresu = twobroadcomp_mod.fit(data_cor,paramsbH,x=l)
        print('El resultado del fit: '+twobroadresu.message)
        lmfit.model.save_modelresult(twobroadresu, path+'broadtwo_modelresult.sav')
        with open(path+'fitbroadtwo_result.txt', 'w') as fh:
            fh.write(twobroadresu.fit_report())

        ################################## Calculate gaussians and final fit #######################################
	# Now we create and plot the individual gaussians of the fit
        b2gaus1 = noOfuncts.gaussian(l,twobroadresu.values['mu_0'],twobroadresu.values['sig_0'],twobroadresu.values['amp_0']) 
        b2gaus2 = noOfuncts.gaussian(l,twobroadresu.values['mu_1'],twobroadresu.values['sig_1'],twobroadresu.values['amp_1'])
        b2gaus3 = noOfuncts.gaussian(l,twobroadresu.values['mu_2'],twobroadresu.values['sig_2'],twobroadresu.values['amp_2']) 
        b2gaus4 = noOfuncts.gaussian(l,twobroadresu.values['mu_3'],twobroadresu.values['sig_3'],twobroadresu.values['amp_3'])
        b2gaus5 = noOfuncts.gaussian(l,twobroadresu.values['mu_4'],twobroadresu.values['sig_4'],twobroadresu.values['amp_4'])
        b2gaus6 = noOfuncts.gaussian(l,twobroadresu.values['mu_20'],twobroadresu.values['sig_20'],twobroadresu.values['amp_20']) 
        b2gaus7 = noOfuncts.gaussian(l,twobroadresu.values['mu_21'],twobroadresu.values['sig_21'],twobroadresu.values['amp_21'])
        b2gaus8 = noOfuncts.gaussian(l,twobroadresu.values['mu_22'],twobroadresu.values['sig_22'],twobroadresu.values['amp_22']) 
        b2gaus9 = noOfuncts.gaussian(l,twobroadresu.values['mu_23'],twobroadresu.values['sig_23'],twobroadresu.values['amp_23'])
        b2gaus10 = noOfuncts.gaussian(l,twobroadresu.values['mu_24'],twobroadresu.values['sig_24'],twobroadresu.values['amp_24'])
        b2gausb = noOfuncts.gaussian(l,twobroadresu.values['mu_b'],twobroadresu.values['sig_b'],twobroadresu.values['amp_b'])
        twobroad_fit = twobroadresu.best_fit

        # We have to calculate the contribution of each component to the global fit
        # Lets define the linear fit data to add to each individual gaussian
        b2gaus_total = twobroad_fit - lin_data_fin
        np.savetxt(path+'fitbroadtwo_best_values.txt',np.c_[twobroadresu.data,twobroadresu.best_fit,lin_data_fin,b2gaus3,b2gaus4,b2gaus5,b2gaus8,b2gaus9,b2gaus10,b2gausb],fmt=('%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f'),header=('Real_data\tBest_fit\tLineal_fit\tNarrow_NII2\tNarrow_Halpha\tNarrow_NII1\tSecond_NII2\tSecond_Halpha\Second_NII1\tBroad_Halpha'))
        # Now lets determine the contribution of the individual components as follows:
        contr_HaN = sum(b2gaus4)
        contr_HaB = sum(b2gausb)
        contr_HaS = sum(b2gaus9)
        ix_Br_sup = np.where(b2gausb > 10**-5)[0][-1]
        ix_Br_inf = np.where(b2gausb > 10**-5)[0][0]
        contr_NII2N = sum(b2gaus5)
        contr_NII1N = sum(b2gaus3)
        contr_NII2S = sum(b2gaus10)
        contr_NII1S = sum(b2gaus8)
        total_flux_NII_Halp = sum(b2gaus_total[ix_Br_inf:ix_Br_sup])

        contr_HaBtoNHa = 100*(contr_HaB/total_flux_NII_Halp)
        contr_HaNtoNHa = 100*(contr_HaN/total_flux_NII_Halp)
        contr_HaStoNHa = 100*(contr_HaS/total_flux_NII_Halp)
        contr_NII2NtoNHa = 100*(contr_NII2N/total_flux_NII_Halp)
        contr_NII2StoNHa = 100*(contr_NII2S/total_flux_NII_Halp)
        contr_NII1NtoNHa = 100*(contr_NII1N/total_flux_NII_Halp)
        contr_NII1StoNHa = 100*(contr_NII1S/total_flux_NII_Halp)

        print('The contribution of the broad component to the total Halpha+N flux is: '+'{:.2f}'.format(contr_HaBtoNHa)+'%')
        np.savetxt(path+'2cBroad_N+Ha_indivcontr.txt',np.c_[contr_NII1NtoNHa,contr_NII1StoNHa,contr_HaNtoNHa,contr_HaStoNHa,contr_HaBtoNHa,contr_NII2NtoNHa,contr_NII2StoNHa],fmt=('%10.7f','%10.7f','%10.7f','%10.7f','%10.7f','%10.7f','%10.7f'),header=('Narrow_NII1(%)\tSecond_NII1(%)\tNarrow_Halpha(%)\tSecond_Halpha(%)\tBroad_Halpha(%)\tNarrow_NII2(%)\tSecond_NII2(%)'))

        # Now we calculate the epsilon under the lines	
        stdb2_s2 = np.std(data_cor[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10]-twobroad_fit[np.where(l_init<l1)[0][-1]:np.where(l_init>l2)[0][0]+10])
        stdb2_s1 = np.std(data_cor[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]]-twobroad_fit[np.where(l_init<l3)[0][-1]-10:np.where(l_init>l4)[0][0]])
        stdb2_n2 = np.std(data_cor[np.where(l_init<l5)[0][-1]:np.where(l_init>l6)[0][0]+10]-twobroad_fit[np.where(l_init<l5)[0][-1]:np.where(l_init>l6)[0][0]+10])
        stdb2_ha = np.std(data_cor[np.where(l_init<l7)[0][-1]:np.where(l_init>l8)[0][0]]-twobroad_fit[np.where(l_init<l7)[0][-1]:np.where(l_init>l8)[0][0]])
        stdb2_n1 = np.std(data_cor[np.where(l_init<l9)[0][-1]-10:np.where(l_init>l10)[0][0]]-twobroad_fit[np.where(l_init<l9)[0][-1]-10:np.where(l_init>l10)[0][0]])
        print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components + Ha is... ')
        print('		For SII2: '+str(stdb2_s2/stadev)+' < 3')
        print('		For SII1: '+str(stdb2_s1/stadev)+' < 3')
        print('		For NII2: '+str(stdb2_n2/stadev)+' < 3')
        print('		For Halp: '+str(stdb2_ha/stadev)+' < 3')
        print('		For NII1: '+str(stdb2_n1/stadev)+' < 3')
    	
        if os.path.exists(path+'eps_adjS2b.txt'): os.remove(path+'eps_adjS2b.txt')
        np.savetxt(path+'eps_adjS2b.txt',np.c_[stdb2_s2/stadev,stdb2_s1/stadev,stdb2_n2/stadev,stdb2_ha/stadev,stdb2_n1/stadev,twobroadresu.chisqr],
    		   ('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('SII2\tSII1\tNII2\tHa\tNII1\tChi2'))

   	# We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
        maxfb2S1 = twobroad_fit[np.where(abs(twobroadresu.values['mu_0']-l)<0.5)[0][0]] #max(twobroad_fit[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]])
        maxfb2S2 = twobroad_fit[np.where(abs(twobroadresu.values['mu_1']-l)<0.5)[0][0]] #max(twobroad_fit[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]])
        maxfb2N1 = twobroad_fit[np.where(abs(twobroadresu.values['mu_2']-l)<0.51)[0][0]] #max(twobroad_fit[np.where(l_init>l9)[0][0]:np.where(l_init<l10)[0][-1]])
        maxfb2Ha = twobroad_fit[np.where(abs(twobroadresu.values['mu_3']-l)<0.5)[0][0]] #max(twobroad_fit[np.where(l_init>l7)[0][0]:np.where(l_init<l8)[0][-1]])
        maxfb2N2 = twobroad_fit[np.where(abs(twobroadresu.values['mu_4']-l)<0.5)[0][0]] #max(twobroad_fit[np.where(l_init>l5)[0][0]:np.where(l_init<l6)[0][-1]])
        # two comps
        vfbS2 = v_luz*((twobroadresu.values['mu_0']-l_SII_2)/l_SII_2)
        vfb2S2 = v_luz*((twobroadresu.values['mu_20']-l_SII_2)/l_SII_2)
        vfb0S2 = v_luz*((twobroadresu.values['mu_b']-l_Halpha)/l_Halpha)
        sigfbS2 = pix_to_v*np.sqrt(twobroadresu.values['sig_0']**2-sig_inst**2)
        sigfb2S2 = pix_to_v*np.sqrt(twobroadresu.values['sig_20']**2-sig_inst**2)
        sigfb0S2 = pix_to_v*np.sqrt(twobroadresu.values['sig_b']**2-sig_inst**2)
        
        if twoSIIresu.params['mu_0'].stderr == None: 
            print('Problem determining the errors! First component ')
            evfbS2,esigfbS2 = 0.,0.
        elif twoSIIresu.params['mu_0'].stderr != None: 
            evfbS2 = ((v_luz/l_SII_2)*twoSIIresu.params['mu_0'].stderr)-er_vsys
        if twoSIIresu.params['sig_0'].stderr == None: 
            print('Problem determining the errors! First component sigma ')
            esigfbS2 = 0.
        elif twoSIIresu.params['sig_0'].stderr != None: 
            esigfbS2 = 47*np.sqrt(twobroadresu.values['sig_0']*twoSIIresu.params['sig_0'].stderr)/(np.sqrt(twobroadresu.values['sig_0']**2-sig_inst**2))

        if twoSIIresu.params['mu_20'].stderr == None:
            print('Problem determining the errors! Second component ')
            evfb2S2 = 0.
        elif twoSIIresu.params['mu_20'].stderr != None:
            evfb2S2 = ((v_luz/l_SII_2)*twoSIIresu.params['mu_20'].stderr)-er_vsys
        if twoSIIresu.params['sig_20'].stderr == None:
            print('Problem determining the errors! Second component sigma ')
            esigfb2S2 = 0.
        elif twoSIIresu.params['sig_20'].stderr != None:
            esigfb2S2 = 47*np.sqrt(twobroadresu.values['sig_20']*twoSIIresu.params['sig_20'].stderr)/(np.sqrt(twobroadresu.values['sig_20']**2-sig_inst**2))
	
        if twobroadresu.params['mu_b'].stderr == None:
            print('Problem determining the errors! Broad component ')
            evfb0S2 = 0.
        elif twobroadresu.params['mu_b'].stderr != None:
            evfb0S2 = ((v_luz/l_SII_2)*twobroadresu.params['mu_b'].stderr)-er_vsys
        if twobroadresu.params['sig_b'].stderr == None:
            print('Problem determining the errors! Broad component sigma ')
            esigfb0S2 = 0.
        else:
            esigfb0S2 = 47*np.sqrt(twobroadresu.values['sig_b']*twobroadresu.params['sig_b'].stderr)/(np.sqrt(twobroadresu.values['sig_b']**2-sig_inst**2))
	
        if os.path.exists(path+'v_sig_adjS2b.txt'): os.remove(path+'v_sig_adjS2b.txt')
        np.savetxt(path+'v_sig_adjS2b.txt',np.c_[vfbS2,evfbS2,vfb2S2,evfb2S2,vfb0S2,evfb0S2,sigfbS2,esigfbS2,sigfb2S2,esigfb2S2,sigfb0S2,esigfb0S2],
    		  ('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('v_SII2\tev_SII2\tv_2SII2\tev_2SII2\tv_bSII2\tev_bSII2\tsig_SII2\tesig_SII2\tsig_2SII2\tesig_2SII2\tsig_bSII2\tesig_bSII2'))
        '''	
	#############################################################################################################
	# We make an F-test to see if it is significant the presence of a broad component in the lines. 
        pre_x = data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-fin2_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]
        pre_y = data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-twobroad_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]
        tx, ty = stats.obrientransform(pre_x, pre_y)
        fvalue1, pvalue1 = stats.f_oneway(tx,ty)
        fb2value, pb2value = stats.f_oneway(data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-fin2_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20],
					    data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-twobroad_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20])
        statvalue2, pb2value2 = stats.levene(data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-fin2_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20],
					     data_cor[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20]-twobroad_fit[np.where(l_init<l9)[0][-1]-20:np.where(l_init>l6)[0][0]+20])
#        fstat = ftest(twocompresu.chisqr,twobroadresu.chisqr,twocompresu.nfree,twobroadresu.nfree)
        print('')
#        print('The probability of a third component (two component vs two + broad Halpha components) in this spectra with the F-test of IDL is: '+str(fstat['p-value']))
        print('The probability of a third component (two component vs two + broad Halpha components) in this spectra with the F-test is: '+str(pb2value))
        print('The probability of a second component (one component vs one + broad Halpha components) in this spectra with the F-test (and O Brien) is: '+str(pvalue1))
        print('The probability of a third component (two component vs two + broad Halpha components) in this spectra with the Levene-test is: '+str(pb2value2))
        print('')
        '''
   	################################################## PLOT #####################################################
        plt.close('all')
    	# MAIN plot
        fig1   = plt.figure(1,figsize=(11, 10))
        frame1 = fig1.add_axes((.1,.25,.85,.65)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
        plt.plot(l,data_cor,'k')			     # Initial data
        plt.plot(l,twobroad_fit,c='red',linestyle='-')
        plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),c='y',linestyle=(0, (5, 8)),label='Linear fit')
        plt.plot(l,b2gaus1,c='b',linestyle='-')
        plt.plot(l,b2gaus2,c='b',linestyle='-')
        plt.plot(l,b2gaus3,c='b',linestyle='-')
        plt.plot(l,b2gaus4,c='b',linestyle='-')
        plt.plot(l,b2gaus5,c='b',linestyle='-',label='Narrow component')
        plt.plot(l,b2gaus6,c='darkviolet',linestyle='-')
        plt.plot(l,b2gaus7,c='darkviolet',linestyle='-')
        plt.plot(l,b2gaus8,c='darkviolet',linestyle='-')
        plt.plot(l,b2gaus9,c='darkviolet',linestyle='-')
        plt.plot(l,b2gaus10,c='darkviolet',linestyle='-',label='Secondary component')
        plt.plot(l,b2gausb,c='darkorange',linestyle='-',label='Broad component')
        textstr = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vfbS2,evfbS2),
			r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vfb2S2,evfb2S2),
			r'$V_{SII_{2-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(vfb0S2,evfb0S2),
		    	r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigfbS2,esigfbS2),
		    	r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigfb2S2,esigfb2S2),
		    	r'$\sigma_{SII_{2-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(sigfb0S2,esigfb0S2),
		        r'$\frac{F_{SII_{2}}}{F_{SII_{1}}}$ = '+ '{:.3f}'.format(maxfb2S2/maxfb2S1),
		    	r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxfb2Ha)+' $10^{-14}$'))

        plt.plot(l[std0:std1],data_cor[std0:std1],'green')	# Zone where the stddev is calculated
        frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
        plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=17)
        plt.tick_params(axis='both', labelsize=15)
        plt.xlim(l[0],l[-1])
        if l[0]<l_OI_1:
            plt.axvline(l_OI_1,0,maxfb2Ha*2+0.1,color='peachpuff')
            plt.axvline(l_OI_2,0,maxfb2Ha*2+0.1,color='peachpuff')
        elif l[0]<l_OI_2 and l[0]>l_OI_1: plt.axvline(l_OI_2,0,maxfb2Ha*2+1,color='peachpuff')
        plt.legend(loc='best',fontsize='large')

    	# RESIDUAL plot
        frame2 = fig1.add_axes((.1,.1,.85,.15))
        plt.plot(l,data_cor-twobroad_fit,'k')		# Main
        plt.xlabel('Wavelength ($\AA$)',fontsize=17)
        plt.ylabel('Residuals',fontsize=17)
        plt.tick_params(axis='both', labelsize=15)
        plt.xlim(l[0],l[-1])
        plt.plot(l,np.zeros(len(l)),color='grey',linestyle='--')         	# Line around zero
        plt.plot(l,np.zeros(len(l))+2*stadev,color='grey',linestyle='--')	# 3 sigma upper limit
        plt.plot(l,np.zeros(len(l))-2*stadev,color='grey',linestyle='--') 	# 3 sigma down limit
        if l[0]<l_OI_1:
            plt.axvline(l_OI_1,-3,3,color='peachpuff')#,linestyle='--')
            plt.axvline(l_OI_2,-3,3,color='peachpuff')
        elif l[0]<l_OI_2 and l[0]>l_OI_1: plt.axvline(l_OI_2,-3,3,color='peachpuff')
        plt.ylim(-(3*stadev)*3,(3*stadev)*3)

        plt.savefig(path+'adj_metS_SII_full_2comp_broadH.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        frame1.text(x_frame,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
        plt.savefig(path+'adj_metS_SII_full_2comp_broadH.png',bbox_inches='tight',pad_inches=0.2)

else: 
    print('Please use "Y" or "N"')

