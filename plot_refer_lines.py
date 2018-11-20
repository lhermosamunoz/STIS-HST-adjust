import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from PyAstronomy.pyasl import ftest
import Ofuncts

def refer_plot(path,l,data_cor,meth,linresu,oneresu,tworesu,l1,l2,l3,l4,l11,l12,l13,l14,std0,std1,z,erz):
	'''
	It gives the plots for one and two components in the reference lines SII and OI

	The parameters needed are:
	path:      Path to the data
	l:         Wavelength range
	data_cor:  Flux for each wavelength
	meth:      Method to be applied (S/O)
	linresu:   Result of the linear fit of the spectra
	oneresu:   Result of the linear+gaussian fit for the reference lines with one component
	tworesu:   Result of the linear+gaussian fit for the reference lines with two components
	l1-l14:    Parts of the spectra where the lines are located
	std0/std1: Where the standard deviation of the continuum is calculated
	z/erz:      Redshift of the galaxy and its error
	'''
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
	sig_inst = 1.32	# pix

	# Systemic velocity of the galaxy
	vsys = v_luz*z
	er_vsys = v_luz*erz

	# Parameters of the linear fit and the std of the continuum	
	new_slop = linresu.values['slope']
	new_intc = linresu.values['intc']
	stadev = np.std(data_cor[std0:std1])
	
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
	    std_2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]]-onefin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]])
	    std_1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]]-onefin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]])
	    ep_1 = std_1/stadev
	    ep_2 = std_2/stadev
	    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
	    print('	For the SII2 line: '+str(ep_2)+' < 3')
	    print('	For the SII1 line: '+str(ep_1)+' < 3')
	    # two components
	    std2_2 = np.std(data_cor[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]]-twofin_fit[np.where(l<l1)[0][-1]:np.where(l>l2)[0][0]])
	    std2_1 = np.std(data_cor[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]]-twofin_fit[np.where(l<l3)[0][-1]:np.where(l>l4)[0][0]])
	    ep2_1 = std2_1/stadev
	    ep2_2 = std2_2/stadev
	    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
	    print('	For the SII2 line: '+str(ep2_2)+' < 3')
	    print('	For the SII1 line: '+str(ep2_1)+' < 3')

	    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
	    maxS1 = max(onefin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
	    maxS2 = max(onefin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
	    max2S1 = max(twofin_fit[np.where(l>l3)[0][0]:np.where(l<l4)[0][-1]])
	    max2S2 = max(twofin_fit[np.where(l>l1)[0][0]:np.where(l<l2)[0][-1]])
	    # one component
	    vS2 = (v_luz*((oneresu.values['mu_0']-l_SII_2)/l_SII_2))-vsys
	    sigS2 = 47*np.sqrt(oneresu.values['sig_0']**2-sig_inst**2)
	    # two comps
	    v2S2 = (v_luz*((tworesu.values['mu_0']-l_SII_2)/l_SII_2))-vsys
	    v20S2 = (v_luz*((tworesu.values['mu_20']-l_SII_2)/l_SII_2))-vsys
	    sig2S2 = 47*np.sqrt(tworesu.values['sig_0']**2-sig_inst**2)
	    sig20S2 = 47*np.sqrt(tworesu.values['sig_20']**2-sig_inst**2)
	    
	    if oneresu.params['mu_0'].stderr == None: 
	        print('Problem determining the errors!')
	        evS2,esigS2 = 0.,0.
	    elif oneresu.params['mu_0'].stderr != None: 
	        evS2 = ((v_luz/l_SII_2)*oneresu.params['mu_0'].stderr)-er_vsys
		esigS2 = 47*np.sqrt(oneresu.values['sig_0']*oneresu.params['sig_0'].stderr)/(np.sqrt(oneresu.values['sig_0']**2-sig_inst**2))

	    if tworesu.params['mu_20'].stderr == None:
	        print('Problem determining the errors!')
	        ev20S2,ev2S2,esig2S2,esig20S2 = 0.,0.,0.,0.
	    elif tworesu.params['mu_20'].stderr != None:
		ev2S2 = ((v_luz/l_SII_2)*tworesu.params['mu_0'].stderr)-er_vsys
		ev20S2 = ((v_luz/l_SII_2)*tworesu.params['mu_20'].stderr)-er_vsys
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
	    std_1 = np.std(data_cor[np.where(l<l11)[0][-1]:np.where(l>l12)[0][0]]-onefin_fit[np.where(l<l11)[0][-1]:np.where(l>l12)[0][0]])
	    std_2 = np.std(data_cor[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]]-onefin_fit[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]])
	    ep_1 = std_1/stadev
	    ep_2 = std_2/stadev
	    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component is... ')
	    print('	For the SII2 line: '+str(ep_2)+' < 3')
	    print('	For the SII1 line: '+str(ep_1)+' < 3')
	    # two components
	    std2_1 = np.std(data_cor[np.where(l<l11)[0][-1]:np.where(l>l12)[0][0]]-twofin_fit[np.where(l<l11)[0][-1]:np.where(l>l12)[0][0]])
	    std2_2 = np.std(data_cor[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]]-twofin_fit[np.where(l<l13)[0][-1]:np.where(l>l14)[0][0]])
	    ep2_1 = std2_1/stadev
	    ep2_2 = std2_2/stadev
	    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 2 components is... ')
	    print('	For the SII2 line: '+str(ep2_2)+' < 3')
	    print('	For the SII1 line: '+str(ep2_1)+' < 3')
	
	    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
	    maxS1 = max(onefin_fit[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]])
	    maxS2 = max(onefin_fit[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]])
	    max2S1 = max(twofin_fit[np.where(l>l13)[0][0]:np.where(l<l14)[0][-1]])
	    max2S2 = max(twofin_fit[np.where(l>l11)[0][0]:np.where(l<l12)[0][-1]])
	    # one component
	    vS2 = (v_luz*((oneresu.values['mu_0']-l_OI_1)/l_OI_1))-vsys
	    sigS2 = 47*np.sqrt(oneresu.values['sig_0']**2-sig_inst**2)
	    # two comps
	    v2S2 = (v_luz*((tworesu.values['mu_0']-l_OI_1)/l_OI_1))-vsys
	    v20S2 = (v_luz*((tworesu.values['mu_20']-l_OI_1)/l_OI_1))-vsys
	    sig2S2 = 47*np.sqrt(tworesu.values['sig_0']**2-sig_inst**2)
	    sig20S2 = 47*np.sqrt(tworesu.values['sig_20']**2-sig_inst**2)

	    if oneresu.params['mu_0'].stderr == None: 
	        print('Problem determining the errors!')
	        evS2,esigS2 = 0.,0.
	    elif oneresu.params['mu_0'].stderr != None: 
	        evS2 = ((v_luz/l_OI_1)*oneresu.params['mu_0'].stderr)-er_vsys
		esigS2 = 47*np.sqrt(oneresu.values['sig_0']*oneresu.params['sig_0'].stderr)/(np.sqrt(oneresu.values['sig_0']**2-sig_inst**2))

	    if tworesu.params['mu_20'].stderr == None:
	        print('Problem determining the errors!')
	        ev20S2, ev2S2, esig2S2, esig20S2 = 0.,0.,0.,0.
	    elif tworesu.params['mu_20'].stderr != None:
	        ev2S2 = ((v_luz/l_OI_1)*tworesu.params['mu_0'].stderr)-er_vsys
		ev20S2 = ((v_luz/l_OI_1)*tworesu.params['mu_20'].stderr)-er_vsys
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
	frame1.text(6350.,max(data_cor), textstr, fontsize=12,verticalalignment='top', bbox=props)
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
	plt.ylim(-3*stadev-0.1,0.1+3*stadev)
	
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
	frame3.text(6350.,max(data_cor), textstr2, fontsize=12,verticalalignment='top', bbox=props)
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
	plt.ylim(-3*stadev-0.1,0.1+3*stadev)
	
	plt.savefig(path+'adj_met'+str(meth)+'_ref_2comp.png')

	##############################################################################################################################################################################
	# We make an F-test to see if it is significant the presence of a second component in the lines. 
	# As the only possible method here is the S-method due to the fact that there are no O-lines in this spectra, 
	# then the method can only be applied to the SII lines (so the wavelength range would be around this two lines)
	fvalue, pvalue = stats.f_oneway(data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-onefin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20],
					data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-twofin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20])
	statist, pvalue2 = stats.levene(data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-onefin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20],
					data_cor[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20]-twofin_fit[np.where(l>l3)[0][0]-20:np.where(l<l2)[0][-1]+20])
	pre_x = data_cor[np.where(l<l3)[0][-1]-20:np.where(l>l2)[0][0]+20]-onefin_fit[np.where(l<l3)[0][-1]-20:np.where(l>l2)[0][0]+20]
	pre_y = data_cor[np.where(l<l3)[0][-1]-20:np.where(l>l2)[0][0]+20]-twofin_fit[np.where(l<l3)[0][-1]-20:np.where(l>l2)[0][0]+20]
	tx, ty = stats.obrientransform(pre_x, pre_y)
	fvalue1, pvalue1 = stats.f_oneway(tx,ty)
	fstat = ftest(oneresu.chisqr,tworesu.chisqr,oneresu.nfree,tworesu.nfree)
	print('')
	print('The probability of a second component (one component vs two components) using the F-test is: '+str(pvalue))
	print('The probability of a second component (one component vs two components) with the F-test (and O Brien) is: '+str(pvalue1))
	print('The probability of a second component (one component vs two components) using the Levene-test is: '+str(pvalue2))
	print('The probability of a second component (one component vs two components) with the F-test of IDL is: '+str(fstat['p-value']))
	print('')

        return ep_1,ep_2,ep2_1,ep2_2

