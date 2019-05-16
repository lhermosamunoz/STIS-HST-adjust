import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import Ofuncts
import os

def broad_plot(path,data_head,l,l_init,data_cor,meth,trigger,linresu,refresu,fullresu,broadresu,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,std0,std1,z,erz):
	'''
	It gives the plots for one and two components + a broad component in the whole spectra

	The parameters needed are:
	path:          Path to the data
	l:             Wavelength range
	data_cor:      Flux for each wavelength
	meth:          Method to be applied (S/O)
	trigger:       This had to be said to the program to decide whether 1 or 2 components
	linresu:       Result of the linear fit of the spectra
	refresu:       Result of the linear+gaussian fit for the reference lines with one component or two components
	fullresu:      Result of the linear+gaussian fit for the spectra with one component or two components
	broadresu:     Result of the linear+gaussian+broad Ha fit for the spectra with one or two components
	l1-l14:        Parts of the spectra where the lines are located
	std0/std1:     Where the standard deviation of the continuum is calculated
	'''
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
	fwhm = 2*np.sqrt(2*np.log(2)) # times sigma
        pix_to_v = 47
	if plate_scale == 0.05078:
	    siginst = 1.1	# A if binning 1x1 // 2.2 if binning 1x2
	    sig_inst = siginst/fwhm
	    ang_to_pix = 0.554
#	    pix_to_v = 25	# km/s
	elif plate_scale == 0.10156:
	    siginst = 2.2
	    sig_inst = siginst/fwhm
	    ang_to_pix = 1.108
#	    pix_to_v = 47	# km/s

	# Systemic velocity of the galaxy
	vsys = v_luz*z
	er_vsys = v_luz*erz
	
	new_slop = linresu.values['slope']
	new_intc = linresu.values['intc']
        lin_data_fin = (linresu.values['slope']*l+linresu.values['intc'])
	stadev = np.std(data_cor[std0:std1])

	if trigger=='Y':
           ################################## Calculate gaussians and final fit #######################################
	    # Now we create and plot the individual gaussians of the fit
	    bgaus1 = Ofuncts.gaussian(l,broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0']) 
	    bgaus2 = Ofuncts.gaussian(l,broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'])
    	    bgaus3 = Ofuncts.gaussian(l,broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2']) 
	    bgaus4 = Ofuncts.gaussian(l,broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'])
    	    bgaus5 = Ofuncts.gaussian(l,broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'])
    	    bgaus6 = Ofuncts.gaussian(l,broadresu.values['mu_5'],broadresu.values['sig_5'],broadresu.values['amp_5'])
    	    bgaus7 = Ofuncts.gaussian(l,broadresu.values['mu_6'],broadresu.values['sig_6'],broadresu.values['amp_6'])
    	    bgaus8 = Ofuncts.gaussian(l,broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])
    	    broad_fit = broadresu.best_fit

            # We have to calculate the contribution of each component to the global fit
            # Lets define the linear fit data to add to each individual gaussian
            bgaus_total = broad_fit - lin_data_fin
            np.savetxt(path+'fitbroad_best_values.txt',np.c_[broadresu.data,broadresu.best_fit,lin_data_fin,bgaus3,bgaus4,bgaus5,bgaus8],fmt=('%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f'),header=('Real_data\tBest_fit\tLineal_fit\tNarrow_NII2\tNarrow_Halpha\tNarrow_NII1\tBroad_Halpha'))
            # Now lets determine the contribution of the individual components as follows:
            contr_HaN = sum(bgaus4)
            contr_HaB = sum(bgaus8)
            ix_Br_sup = np.where(bgaus8 > 10**-5)[0][-1]
            ix_Br_inf = np.where(bgaus8 > 10**-5)[0][0]
            contr_NII2N = sum(bgaus3)
            contr_NII1N = sum(bgaus5)
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
            stdb_o1 = np.std(data_cor[np.where(l_init<l11)[0][-1]:np.where(l_init>l12)[0][0]+10]-broad_fit[np.where(l_init<l11)[0][-1]:np.where(l_init>l12)[0][0]+10])
            stdb_o2 = np.std(data_cor[np.where(l_init<l13)[0][-1]-10:np.where(l_init>l14)[0][0]+10]-broad_fit[np.where(l_init<l13)[0][-1]-10:np.where(l_init>l14)[0][0]+10])
    	    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component + Ha is... ')
    	    print('		For SII2: '+str(stdb_s2/stadev)+' < 3')
    	    print('		For SII1: '+str(stdb_s1/stadev)+' < 3')
    	    print('		For NII2: '+str(stdb_n2/stadev)+' < 3')
    	    print('		For Halp: '+str(stdb_ha/stadev)+' < 3')
    	    print('		For NII1: '+str(stdb_n1/stadev)+' < 3')
            print('             For OI1: '+str(stdb_o1/stadev)+' < 3')
            print('             For OI2: '+str(stdb_o2/stadev)+' < 3')
    	    
    	    if os.path.exists(path+'eps_adj'+str(meth)+'_1b.txt'): os.remove(path+'eps_adj'+str(meth)+'_1b.txt')
    	    np.savetxt(path+'eps_adj'+str(meth)+'_1b.txt',np.c_[stdb_s2/stadev,stdb_s1/stadev,stdb_n2/stadev,stdb_ha/stadev,stdb_n1/stadev,stdb_o2/stadev,stdb_o1/stadev,broadresu.chisqr],('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('SII2\tSII1\tNII2\tHa\tNII1\tOI2\tOI1\tChi2'))

   	    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
    	    try:
                maxbS1 = broad_fit[np.where(abs(broadresu.values['mu_0']-l)<0.3)[0][0]] #max(broad_fit[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]])
                maxbS2 = broad_fit[np.where(abs(broadresu.values['mu_1']-l)<0.3)[0][0]] #max(broad_fit[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]])
                maxbN1 = broad_fit[np.where(abs(broadresu.values['mu_2']-l)<0.3)[0][0]] #max(broad_fit[np.where(l_init>l9)[0][0]:np.where(l_init<l10)[0][-1]])
                maxbHa = broad_fit[np.where(abs(broadresu.values['mu_3']-l)<0.3)[0][0]] #max(broad_fit[np.where(l_init>l7)[0][0]:np.where(l_init<l8)[0][-1]])
                maxbN2 = broad_fit[np.where(abs(broadresu.values['mu_4']-l)<0.3)[0][0]] #max(broad_fit[np.where(l_init>l5)[0][0]:np.where(l_init<l6)[0][-1]])
                maxbO2 = broad_fit[np.where(abs(broadresu.values['mu_6']-l)<0.3)[0][0]] #max(broad_fit[np.where(l_init>l13)[0][0]:np.where(l_init<l14)[0][-1]])
                maxbO1 = broad_fit[np.where(abs(broadresu.values['mu_5']-l)<0.3)[0][0]] #max(broad_fit[np.where(l_init>l11)[0][0]:np.where(l_init<l12)[0][-1]])
            except IndexError:
                print('ERROR: index out of range. Setting the flux values of the OI 1 line to 0.')
                maxb01 = 0
	    # one component + Halpha
            sigbS2 = pix_to_v*np.sqrt(broadresu.values['sig_0']**2-sig_inst**2)
            sigb0S2 = pix_to_v*np.sqrt(broadresu.values['sig_b']**2-sig_inst**2)
            if refresu.params['sig_0'].stderr == None:
                esigbS2 = 0.
            else: 
		esigbS2 = pix_to_v*np.sqrt(broadresu.values['sig_0']*refresu.params['sig_0'].stderr)/(np.sqrt(broadresu.values['sig_0']**2-sig_inst**2))
            if broadresu.params['sig_b'].stderr == None:
                esigb0S2 = 0.
            else: 
		esigb0S2 = pix_to_v*np.sqrt(broadresu.values['sig_b']*broadresu.params['sig_b'].stderr)/(np.sqrt(broadresu.values['sig_b']**2-sig_inst**2))

	    if meth == 'S':
		vS2 = v_luz*((broadresu.values['mu_0']-l_SII_2)/l_SII_2)
		vbS2 = v_luz*((broadresu.values['mu_b']-l_Halpha)/l_Halpha)
		if refresu.params['mu_0'].stderr == None: 
		    print('Problem determining the errors! First component ')
		    evS2 = 0.
		elif refresu.params['mu_0'].stderr != None: 
		    evS2 = ((v_luz/l_SII_2)*refresu.params['mu_0'].stderr)-er_vsys
		if broadresu.params['mu_b'].stderr == None:
		    evbS2 = 0.
		else:
		    evbS2 = ((v_luz/l_Halpha)*broadresu.params['mu_b'].stderr)-er_vsys
    	        textstr = '\n'.join((r'$V_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
			r'$V_{H_{\alpha-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(vbS2,evbS2),
		    	r'$\sigma_{SII_{2}}$ = '+ '{:.2f} +- {:.2f}'.format(sigbS2,esigbS2),
		    	r'$\sigma_{H_{\alpha-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(sigb0S2,esigb0S2),
			r'$F_{SII_{2}}/F_{SII_{1}}$ = '+ '{:.3f}'.format(maxbS2/maxbS1),
			r'$F_{NII_{2}}/F_{NII_{1}}$ = '+ '{:.3f}'.format(maxbN2/maxbN1),
		    	r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxbHa)+' $10^{-14}$',
			r'$F_{OI_{2}}/F_{OI_{1}}$ = '+ '{:.3f}'.format(maxbO2/maxbO1)))

	    elif meth == 'O':
		vS2 = v_luz*((broadresu.values['mu_5']-l_OI_1)/l_OI_1)
		vbS2 = v_luz*((broadresu.values['mu_b']-l_Halpha)/l_Halpha)
		if refresu.params['mu_0'].stderr == None: 
		    print('Problem determining the errors! First component ')
		    evS2 = 0.
		elif refresu.params['mu_0'].stderr != None: 
		    evS2 = ((v_luz/l_OI_1)*refresu.params['mu_0'].stderr)-er_vsys
		if broadresu.params['mu_b'].stderr == None:
		    evbS2 = 0.
		else:
		    evbS2 = ((v_luz/l_Halpha)*broadresu.params['mu_b'].stderr)-er_vsys

    	        textstr = '\n'.join((r'$V_{OI_{1}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
			r'$V_{H_{\alpha-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(vbS2,evbS2),
		    	r'$\sigma_{OI_{1}}$ = '+ '{:.2f} +- {:.2f}'.format(sigbS2,esigbS2),
		    	r'$\sigma_{H_{\alpha-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(sigb0S2,esigb0S2),
			r'$F_{SII_{2}}/F_{SII_{1}}$ = '+ '{:.3f}'.format(maxbS2/maxbS1),
			r'$F_{NII_{2}}/F_{NII_{1}}$ = '+ '{:.3f}'.format(maxbN2/maxbN1),
		    	r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxbHa)+' $10^{-14}$',
			r'$F_{OI_{2}}/F_{OI_{1}}$ = '+ '{:.3f}'.format(maxbO2/maxbO1)))

	    # Save the velocity and sigma for all components
	    if os.path.exists(path+'v_sig_adj'+str(meth)+'_1b.txt'): os.remove(path+'v_sig_adj'+str(meth)+'_1b.txt')
	    np.savetxt(path+'v_sig_adj'+str(meth)+'_1b.txt',np.c_[vS2,evS2,vbS2,evbS2,sigbS2,esigbS2,sigb0S2,esigb0S2],
    		  ('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('v_ref\tev_ref\tv_broadH\tev_broadH\tsig_ref\tesig_ref\tsig_broadH\tesig_broadH'))

   	    ################################################ PLOT ######################################################
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
    	    plt.plot(l,bgaus5,'b-')
    	    plt.plot(l,bgaus6,'b-')
    	    plt.plot(l,bgaus7,'b-',label='Narrow component')
    	    plt.plot(l,bgaus8,c='darkorange',linestyle='-',label='Broad component')

    	    plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
    	    frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
    	    plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=17)
    	    plt.tick_params(axis='both', labelsize=15)
    	    plt.xlim(l[0],l[-1])
    	    plt.legend(loc='best',fontsize='large')

    	    # RESIDUAL plot
    	    frame2 = fig1.add_axes((.1,.1,.85,.15))
    	    plt.plot(l,data_cor-broad_fit,c='k')		# Main
    	    plt.xlabel('Wavelength ($\AA$)',fontsize=17)
    	    plt.ylabel('Residuals',fontsize=17)
    	    plt.tick_params(axis='both', labelsize=15)
    	    plt.xlim(l[0],l[-1])
    	    plt.plot(l,np.zeros(len(l)),c='grey',linestyle='--')         	# Line around zero
    	    plt.plot(l,np.zeros(len(l))+2*stadev,c='grey',linestyle='--')	# 3 sigma upper limit
    	    plt.plot(l,np.zeros(len(l))-2*stadev,c='grey',linestyle='--') 	# 3 sigma down limit
    	    plt.ylim(-(3*stadev)*3,(3*stadev)*3)

    	    plt.savefig(path+'adj_met'+str(meth)+'_full_1comp_broadH.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)
    	    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    	    frame1.text(6350.,max(data_cor)+0.2, textstr, fontsize=12,verticalalignment='top', bbox=props)
    	    plt.savefig(path+'adj_met'+str(meth)+'_full_1comp_broadH.png',bbox_inches='tight',pad_inches=0.2)

###################################################################################################################################################################################################

	elif trigger=='N':
	    ################################## Calculate gaussians and final fit #######################################
	    # Now we create and plot the individual gaussians of the fit
	    b2gaus1 = Ofuncts.gaussian(l,broadresu.values['mu_0'],broadresu.values['sig_0'],broadresu.values['amp_0']) 
	    b2gaus2 = Ofuncts.gaussian(l,broadresu.values['mu_1'],broadresu.values['sig_1'],broadresu.values['amp_1'])
	    b2gaus3 = Ofuncts.gaussian(l,broadresu.values['mu_2'],broadresu.values['sig_2'],broadresu.values['amp_2']) 
	    b2gaus4 = Ofuncts.gaussian(l,broadresu.values['mu_3'],broadresu.values['sig_3'],broadresu.values['amp_3'])
	    b2gaus5 = Ofuncts.gaussian(l,broadresu.values['mu_4'],broadresu.values['sig_4'],broadresu.values['amp_4'])
    	    b2gaus6 = Ofuncts.gaussian(l,broadresu.values['mu_5'],broadresu.values['sig_5'],broadresu.values['amp_5'])
    	    b2gaus7 = Ofuncts.gaussian(l,broadresu.values['mu_6'],broadresu.values['sig_6'],broadresu.values['amp_6'])
	    b2gaus8 = Ofuncts.gaussian(l,broadresu.values['mu_20'],broadresu.values['sig_20'],broadresu.values['amp_20']) 
	    b2gaus9 = Ofuncts.gaussian(l,broadresu.values['mu_21'],broadresu.values['sig_21'],broadresu.values['amp_21'])
	    b2gaus10 = Ofuncts.gaussian(l,broadresu.values['mu_22'],broadresu.values['sig_22'],broadresu.values['amp_22']) 
	    b2gaus11 = Ofuncts.gaussian(l,broadresu.values['mu_23'],broadresu.values['sig_23'],broadresu.values['amp_23'])
	    b2gaus12 = Ofuncts.gaussian(l,broadresu.values['mu_24'],broadresu.values['sig_24'],broadresu.values['amp_24'])
	    b2gaus13 = Ofuncts.gaussian(l,broadresu.values['mu_25'],broadresu.values['sig_25'],broadresu.values['amp_25'])
	    b2gaus14 = Ofuncts.gaussian(l,broadresu.values['mu_26'],broadresu.values['sig_26'],broadresu.values['amp_26'])
	    b2gausb = Ofuncts.gaussian(l,broadresu.values['mu_b'],broadresu.values['sig_b'],broadresu.values['amp_b'])
	    twobroad_fit = broadresu.best_fit

            # We have to calculate the contribution of each component to the global fit
            # Lets define the linear fit data to add to each individual gaussian
            b2gaus_total = twobroad_fit - lin_data_fin
            np.savetxt(path+'fitbroadtwo_best_values.txt',np.c_[broadresu.data,broadresu.best_fit,lin_data_fin,b2gaus3,b2gaus4,b2gaus5,b2gaus10,b2gaus11,b2gaus12,b2gausb],fmt=('%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f','%5.10f'),header=('Real_data\tBest_fit\tLineal_fit\tNarrow_NII2\tNarrow_Halpha\tNarrow_NII1\tSecond_NII2\tSecond_Halpha\Second_NII1\tBroad_Halpha'))
            # Now lets determine the contribution of the individual components as follows:
            contr_HaN = sum(b2gaus4)
            contr_HaB = sum(b2gausb)
            contr_HaS = sum(b2gaus11)
            ix_Br_sup = np.where(b2gausb > 10**-5)[0][-1]
            ix_Br_inf = np.where(b2gausb > 10**-5)[0][0]
            contr_NII2N = sum(b2gaus3)
            contr_NII1N = sum(b2gaus5)
            contr_NII2S = sum(b2gaus10)
            contr_NII1S = sum(b2gaus12)
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
            stdb2_o1 = np.std(data_cor[np.where(l_init<l11)[0][-1]:np.where(l_init>l12)[0][0]+10]-twobroad_fit[np.where(l_init<l11)[0][-1]:np.where(l_init>l12)[0][0]+10])
            stdb2_o2 = np.std(data_cor[np.where(l_init<l13)[0][-1]-10:np.where(l_init>l14)[0][0]+10]-twobroad_fit[np.where(l_init<l13)[0][-1]-10:np.where(l_init>l14)[0][0]+10])

	    print('The condition for each line (in the same order as before) needs to be std_line < 3*std_cont --> for 1 component + Ha is... ')
	    print('		For SII2: '+str(stdb2_s2/stadev)+' < 3')
	    print('		For SII1: '+str(stdb2_s1/stadev)+' < 3')
	    print('		For NII2: '+str(stdb2_n2/stadev)+' < 3')
	    print('		For Halp: '+str(stdb2_ha/stadev)+' < 3')
	    print('		For NII1: '+str(stdb2_n1/stadev)+' < 3')
            print('             For OI2: '+str(stdb2_o2/stadev)+' < 3')
            print('             For OI1: '+str(stdb2_o1/stadev)+' < 3')
	    
	    if os.path.exists(path+'eps_adj'+str(meth)+'2b.txt'): os.remove(path+'eps_adj'+str(meth)+'2b.txt')
    	    np.savetxt(path+'eps_adj'+str(meth)+'2b.txt',np.c_[stdb2_s2/stadev,stdb2_s1/stadev,stdb2_n2/stadev,stdb2_ha/stadev,stdb2_n1/stadev,stdb2_o2/stadev,stdb2_o1/stadev,broadresu.chisqr],('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),header=('SII2\tSII1\tNII2\tHa\tNII1\tOI2\tOI1\tChi2'))

	    # We determine the maximum flux of the fit for all the lines, and the velocity and sigma components
	    try:
                maxfbS1 = twobroad_fit[np.where(abs(broadresu.values['mu_0']-l)<0.3)[0][0]] #max(twobroad_fit[np.where(l_init>l3)[0][0]:np.where(l_init<l4)[0][-1]])
                maxfbS2 = twobroad_fit[np.where(abs(broadresu.values['mu_1']-l)<0.3)[0][0]] #max(twobroad_fit[np.where(l_init>l1)[0][0]:np.where(l_init<l2)[0][-1]])
                maxfbN1 = twobroad_fit[np.where(abs(broadresu.values['mu_2']-l)<0.3)[0][0]] #max(twobroad_fit[np.where(l_init>l9)[0][0]:np.where(l_init<l10)[0][-1]])
                maxfbHa = twobroad_fit[np.where(abs(broadresu.values['mu_3']-l)<0.3)[0][0]] #max(twobroad_fit[np.where(l_init>l7)[0][0]:np.where(l_init<l8)[0][-1]])
                maxfbN2 = twobroad_fit[np.where(abs(broadresu.values['mu_4']-l)<0.3)[0][0]] #max(twobroad_fit[np.where(l_init>l5)[0][0]:np.where(l_init<l6)[0][-1]])
                maxfbO2 = twobroad_fit[np.where(abs(broadresu.values['mu_6']-l)<0.3)[0][0]] #max(twobroad_fit[np.where(l_init>l13)[0][0]:np.where(l_init<l14)[0][-1]])
                maxfbO1 = twobroad_fit[np.where(abs(broadresu.values['mu_5']-l)<0.3)[0][0]] #max(twobroad_fit[np.where(l_init>l11)[0][0]:np.where(l_init<l12)[0][-1]])
            except IndexError:
                if broadresu.values['mu_5']<l[0]:
                    print('ERROR: index out of range. Setting the flux values of the OI 1 line to 0.')
                    maxfbO1 = 0.
                elif broadresu.values['mu_0']>l[-1]:
                    print('ERROR: index out of range. Setting the flux values of the SI 1 line to 0.')
                    maxfbS1 = 0.

	    # two comps + Halpha
	    sigS2 = pix_to_v*np.sqrt(broadresu.values['sig_0']**2-sig_inst**2)
	    sig2S2 = pix_to_v*np.sqrt(broadresu.values['sig_20']**2-sig_inst**2)
	    sigbS2 = pix_to_v*np.sqrt(broadresu.values['sig_b']**2-sig_inst**2)
            if refresu.params['sig_0'].stderr == None and refresu.params['sig_20'].stderr == None:
                esigS2,esig2S2 = 0.,0.
            else: 
		esigS2 = pix_to_v*np.sqrt(broadresu.values['sig_0']*refresu.params['sig_0'].stderr)/(np.sqrt(broadresu.values['sig_0']**2-sig_inst**2))
		esig2S2 = pix_to_v*np.sqrt(broadresu.values['sig_20']*refresu.params['sig_20'].stderr)/(np.sqrt(broadresu.values['sig_20']**2-sig_inst**2))
            if broadresu.params['sig_b'].stderr == None:
                esigbS2 = 0.
            else: 
		esigbS2 = pix_to_v*np.sqrt(broadresu.values['sig_b']*broadresu.params['sig_b'].stderr)/(np.sqrt(broadresu.values['sig_b']**2-sig_inst**2))

	    if meth == 'S':
		vS2 = v_luz*((broadresu.values['mu_0']-l_SII_2)/l_SII_2)
		v2S2 = v_luz*((broadresu.values['mu_20']-l_SII_2)/l_SII_2)
		vbS2 = v_luz*((broadresu.values['mu_b']-l_Halpha)/l_Halpha)
		if refresu.params['mu_0'].stderr == None: 
		    print('Problem determining the errors! First component ')
		    evS2,ev2S2 = 0.,0.
		elif refresu.params['mu_0'].stderr != None: 
		    evS2 = ((v_luz/l_SII_2)*refresu.params['mu_0'].stderr)-er_vsys
		    ev2S2 = ((v_luz/l_SII_2)*refresu.params['mu_20'].stderr)-er_vsys
		if broadresu.params['mu_b'].stderr == None:
		    evbS2 = 0.
		else:
		    evbS2 = ((v_luz/l_Halpha)*broadresu.params['mu_b'].stderr)-er_vsys
		
    	        textstr = '\n'.join((r'$V_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
			r'$V_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2S2,ev2S2),
			r'$V_{H_{\alpha-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(vbS2,evbS2),
		    	r'$\sigma_{SII_{2-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    	r'$\sigma_{SII_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2S2,esig2S2),
		    	r'$\sigma_{H_{\alpha-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(sigbS2,esigbS2),
			r'$F_{SII_{2}}/F_{SII_{1}}$ = '+ '{:.3f}'.format(maxfbS2/maxfbS1),
			r'$F_{NII_{2}}/F_{NII_{1}}$ = '+ '{:.3f}'.format(maxfbN2/maxfbN1),
		    	r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxfbHa)+' $10^{-14}$',
			r'$F_{OI_{2}}/F_{OI_{1}}$ = '+ '{:.3f}'.format(maxfbO2/maxfbO1)))

	    elif meth == 'O':
		vS2 = v_luz*((broadresu.values['mu_5']-l_OI_1)/l_OI_1)
		v2S2 = v_luz*((broadresu.values['mu_25']-l_OI_1)/l_OI_1)
		vbS2 = v_luz*((broadresu.values['mu_b']-l_Halpha)/l_Halpha)
		if refresu.params['mu_0'].stderr == None: 
		    print('Problem determining the errors! First component ')
		    evS2,ev2S2 = 0.,0.
		elif refresu.params['mu_0'].stderr != None: 
		    evS2 = ((v_luz/l_OI_1)*refresu.params['mu_0'].stderr)-er_vsys
		    ev2S2 = ((v_luz/l_OI_1)*refresu.params['mu_20'].stderr)-er_vsys
		if broadresu.params['mu_b'].stderr == None:
		    evbS2 = 0.
		else:
		    evbS2 = ((v_luz/l_Halpha)*broadresu.params['mu_b'].stderr)-er_vsys
    	        textstr = '\n'.join((r'$V_{OI_{1-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(vS2,evS2),
			r'$V_{OI_{2-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(v2S2,ev2S2),
			r'$V_{H_{\alpha-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(vbS2,evbS2),
		    	r'$\sigma_{OI_{1-1comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sigS2,esigS2),
		    	r'$\sigma_{OI_{1-2comp}}$ = '+ '{:.2f} +- {:.2f}'.format(sig2S2,esig2S2),
		    	r'$\sigma_{H_{\alpha-broad}}$ = '+ '{:.2f} +- {:.2f}'.format(sigbS2,esigbS2),
			r'$F_{SII_{2}}/F_{SII_{1}}$ = '+ '{:.3f}'.format(maxfbS2/maxfbS1),
			r'$F_{NII_{2}}/F_{NII_{1}}$ = '+ '{:.3f}'.format(maxfbN2/maxfbN1),
		    	r'$F_{H_{\alpha}}$ = '+ '{:.3f}'.format(maxfbHa)+' $10^{-14}$',
			r'$F_{OI_{2}}/F_{OI_{1}}$ = '+ '{:.3f}'.format(maxfbO2/maxfbO1)))

	    # Save the velocity and sigma for all components
	    if os.path.exists(path+'v_sig_adj'+str(meth)+'_2b.txt'): os.remove(path+'v_sig_adj'+str(meth)+'_2b.txt')
	    np.savetxt(path+'v_sig_adj'+str(meth)+'_2b.txt',np.c_[vS2,evS2,v2S2,ev2S2,vbS2,evbS2,sigS2,esigS2,sig2S2,esig2S2,sigbS2,esigbS2],
			('%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f','%8.5f'),
			header=('v_ref2\tev_ref2\tv_2ref2\tev_2ref2\tv_broadH\tev_broadH\tsig_ref2\tesig_ref2\tsig_2ref2\tesig_2ref2\tsig_broadH\tesig_broadH'))

	    ################################################ PLOT ######################################################
	    plt.close('all')
	    # MAIN plot
	    fig1   = plt.figure(1,figsize=(11, 10))
	    frame1 = fig1.add_axes((.1,.25,.85,.65)) 	     # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
	    plt.plot(l,data_cor,'k')			     # Initial data
	    plt.plot(l,twobroad_fit,'r-')
	    plt.plot(l,(linresu.values['slope']*l+linresu.values['intc']),c='y',linestyle=(0, (5, 8)),label='Linear fit')
	    plt.plot(l,b2gaus1,'b-')
	    plt.plot(l,b2gaus2,'b-')
	    plt.plot(l,b2gaus3,'b-')
	    plt.plot(l,b2gaus4,'b-')
	    plt.plot(l,b2gaus5,'b-')
	    plt.plot(l,b2gaus6,'b-')
	    plt.plot(l,b2gaus7,'b-',label='Narrow component')
	    plt.plot(l,b2gaus8,c='darkviolet',linestyle='-')
	    plt.plot(l,b2gaus9,c='darkviolet',linestyle='-')
	    plt.plot(l,b2gaus10,c='darkviolet',linestyle='-')
	    plt.plot(l,b2gaus11,c='darkviolet',linestyle='-')
	    plt.plot(l,b2gaus12,c='darkviolet',linestyle='-')
	    plt.plot(l,b2gaus13,c='darkviolet',linestyle='-')
	    plt.plot(l,b2gaus14,c='darkviolet',linestyle='-',label='Second component')
	    plt.plot(l,b2gausb,c='darkorange',linestyle='-',label='Broad component')

	    plt.plot(l[std0:std1],data_cor[std0:std1],'g')	# Zone where the stddev is calculated
	    frame1.set_xticklabels([]) 			# Remove x-tic labels for the first frame
	    plt.ylabel('Flux (x10$^{-14} \mathrm{erg/s/cm^{2} / \AA}$)',fontsize=17)
	    plt.tick_params(axis='both', labelsize=15)
	    plt.xlim(l[0],l[-1])
	    plt.legend(loc='best',fontsize='large')

    	    # RESIDUAL plot
	    frame2 = fig1.add_axes((.1,.1,.85,.15))
	    plt.plot(l,data_cor-twobroad_fit,c='k')		# Main
	    plt.xlabel('Wavelength ($\AA$)',fontsize=17)
	    plt.ylabel('Residuals',fontsize=17)
	    plt.tick_params(axis='both', labelsize=15)
	    plt.xlim(l[0],l[-1])
	    plt.plot(l,np.zeros(len(l)),c='grey',linestyle='--')         	# Line around zero
	    plt.plot(l,np.zeros(len(l))+2*stadev,c='grey',linestyle='--')	# 3 sigma upper limit
	    plt.plot(l,np.zeros(len(l))-2*stadev,c='grey',linestyle='--') 	# 3 sigma down limit
	    plt.ylim(-(3*stadev)*4,(3*stadev)*4)
	    
	    plt.savefig(path+'adj_met'+str(meth)+'_full_2comp_broadH.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)
	    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
	    frame1.text(6350.,max(data_cor)+0.2, textstr, fontsize=12,verticalalignment='top', bbox=props)
	    plt.savefig(path+'adj_met'+str(meth)+'_full_2comp_broadH.png',bbox_inches='tight',pad_inches=0.2)
