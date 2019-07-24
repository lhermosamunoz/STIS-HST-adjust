'''
Script to calculate line ratios and create BPTs
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -------------------------------------------------------------------------------------------------
# Path to the data 
PalomarPath = '/mnt/data/lhermosa/L2_palomar_stsub/'
HSTPath = '/mnt/data/lhermosa/HLA_data/'
gal = ['NGC2685','NGC3245','NGC4374','NGC4486','NGC4552','NGC4594','NGC4698','NGC4736']
allPalPaths = []
allHSTPaths = []
for i in gal:
   allPalPaths.append(PalomarPath+i+'_results')
   allHSTPaths.append(HSTPath+i+'/S_method/')
   print('Adding '+PalomarPath+i+'_results to parentFolds...')
   print('Adding '+HSTPath+i+'/S_method/ to parentFolds...')

# Read the data with the fluxes
flux_OI = []		# OI lambda 6300 
flux_OI_HST = []	# OI lambda 6300 
flux_SII = []		# SII lambda 
flux_SII_HST = []	# SII lambda 
flux_NII = []		# NII lambda 6584
flux_NII_HST = []	# NII lambda 6584
flux_Halpha = []	# Halpha lambda 6563
flux_Halpha_HST = []	# Halpha lambda 6563
index_gal = 0
ix_gal_HST = 0
for galaxy in allPalPaths:
    # Define the PalomarPath to save the data
    parentFold = PalomarPath+gal[index_gal]+'_results/'
    dataFile = np.genfromtxt(parentFold+gal[index_gal]+'_fluxes_final.txt')
    flux_SII.append(dataFile[0])	# Both lines together!
    flux_NII.append(dataFile[1])	# 6584 AA
    flux_Halpha.append(dataFile[2])	# Halpha
    flux_OI.append(dataFile[5])
    index_gal += 1

for galaxy in allHSTPaths:
    # Define the HSTPath to save the data
    parentFold = HSTPath+gal[ix_gal_HST]+'/S_method/'
    dataFile = np.genfromtxt(parentFold+gal[ix_gal_HST]+'_fluxes_final.txt')
    flux_SII_HST.append(dataFile[0])        # Both lines together!
    flux_NII_HST.append(dataFile[1])        # 6584 AA
    flux_Halpha_HST.append(dataFile[2])     # Halpha
    if len(dataFile) == 4:
        print('No OI available for '+gal[ix_gal_HST]+' in the HST spectra!')
    else: 
        flux_OI_HST.append(dataFile[5])
    ix_gal_HST += 1

flux_OI = np.array(flux_OI)
flux_OI_HST = np.array(flux_OI_HST)
flux_NII = np.array(flux_NII)
flux_NII_HST = np.array(flux_NII_HST)
flux_SII = np.array(flux_SII)
flux_SII_HST = np.array(flux_SII_HST)
flux_Halpha = np.array(flux_Halpha)
flux_Halpha_HST = np.array(flux_Halpha_HST)

#plt.xlabel(r'log([SII]/H$\alpha$)',fontsize=18)
#plt.ylabel(r'log([OI]/H$\beta$)',fontsize=18)
    
# Now we have to calculate the line ratios and plot the empirical separating lines
SII_Halpha = np.log10(flux_SII)
SII_Halpha_HST = np.log10(flux_SII_HST)
NII_Halpha = np.log10(flux_NII)
NII_Halpha_HST = np.log10(flux_NII_HST)
OI_Halpha = np.log10(flux_OI)
OI_Halpha_HST = np.log10(flux_OI_HST)

print('The SII/Halpha ratio for the objects is: ')
print(str(SII_Halpha))
print('The mean SII/Halpha ratio is: '+str(np.mean(SII_Halpha)))
print('The NII/Halpha ratio for the objects is: ')
print(str(NII_Halpha))
print('The mean NII/Halpha ratio is: '+str(np.mean(NII_Halpha)))
print('The OI/Halpha ratio for the objects is: ')
print(str(OI_Halpha))
print('The mean OI/Halpha ratio is: '+str(np.mean(OI_Halpha)))

############################# PLOT ##########################
plt.ion()
plt.show()
fig = plt.figure(figsize=(7,6))
gs = GridSpec(4,4)
ax1 = fig.add_subplot(gs[1:4,0:3])
plt.plot(SII_Halpha,SII_Halpha_HST,'ko',linestyle='None')
plt.fill_between(np.linspace(-0.8,0.8),np.linspace(-0.8,0.8)+0.2,np.linspace(-0.8,0.8)-0.2,color='paleturquoise',alpha=0.5)
plt.fill_between(np.linspace(-0.8,0.8),np.linspace(-0.8,0.8)+0.1,np.linspace(-0.8,0.8)-0.1,color='lightsalmon',alpha=0.5)
plt.plot(np.linspace(-0.8,0.8),np.linspace(-0.8,0.8),c='grey',linestyle='--')
# To include the labels of each galaxy in the plot...
for i in range(len(SII_Halpha)):
    plt.text(SII_Halpha[i]-0.06,SII_Halpha_HST[i]+0.01,gal[i],fontsize=8)
plt.xlim(-0.5,0.6)
plt.ylim(-0.05,0.35)
plt.xlabel(r'log([SII]/H$\mathrm{\alpha}$) (Palomar)',fontsize=16)
plt.ylabel(r'log([SII]/H$\mathrm{\alpha}$) (HST)',fontsize=16)
plt.locator_params(axis='y', nbins=5)
plt.tick_params(axis='both',direction='in',labelsize=14)

# Histogram on the x margin
ax_margx = fig.add_subplot(gs[0,0:3])
plt.hist(SII_Halpha,histtype='stepfilled',color='silver')
plt.hist(SII_Halpha,histtype='step',color='k')
plt.ylim(0,4)
plt.setp(ax_margx.get_xticklabels(), visible=False)
for i in range(len(ax_margx.get_yticklabels())-1):
    plt.setp(ax_margx.get_yticklabels()[i], visible=False)
plt.setp(ax_margx.yaxis.get_minorticklines(), visible=False)
plt.xlim(-0.5,0.6)
plt.tick_params(axis='both',direction='in',labelsize=12)

# Histogram on the y margin 
ax_margy = fig.add_subplot(gs[1:4,3])
plt.hist(SII_Halpha_HST,histtype='stepfilled',color='silver',orientation='horizontal')
plt.hist(SII_Halpha_HST,histtype='step',color='k',orientation='horizontal')
plt.xlim(0,4)
plt.setp(ax_margy.get_yticklabels(), visible=False)
for i in range(len(ax_margy.get_xticklabels())-1):
    plt.setp(ax_margy.get_xticklabels()[i], visible=False)
plt.setp(ax_margy.xaxis.get_minorticklines(), visible=False)
plt.ylim(-0.05,0.35)
plt.locator_params(axis='y', nbins=5)
plt.tick_params(axis='both',direction='in',labelsize=12)

plt.subplots_adjust(wspace=0.,hspace=0.0)

#plt.savefig('/mnt/data/lhermosa/HLA_data/line_ratios/SII_Halpha_HSTvsPalomar_labels.png',bbox_inches='tight',pad_inches=0.2)
#plt.savefig('/mnt/data/lhermosa/HLA_data/line_ratios/SII_Halpha_HSTvsPalomar_labels.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)

# ----------------------------------------
fig2 = plt.figure(figsize=(7,6))
gs2 = GridSpec(4,4)
ax21 = fig2.add_subplot(gs[1:4,0:3])
plt.plot(NII_Halpha,NII_Halpha_HST,'ko',linestyle='None')
plt.fill_between(np.linspace(-0.6,0.6),np.linspace(-0.6,0.6)+0.2,np.linspace(-0.6,0.6)-0.2,color='paleturquoise',alpha=0.5)
plt.fill_between(np.linspace(-0.6,0.6),np.linspace(-0.6,0.6)+0.1,np.linspace(-0.6,0.6)-0.1,color='lightsalmon',alpha=0.5)
plt.plot(np.linspace(-0.6,0.6),np.linspace(-0.6,0.6),c='grey',linestyle='--')
# To include the labels of each galaxy in the plot...
for i in range(len(SII_Halpha)):
    plt.text(NII_Halpha[i]-0.03,NII_Halpha_HST[i]+0.01,gal[i],fontsize=8)
plt.locator_params(axis='y', nbins=5)
plt.xlim(-0.6,0.3)
plt.ylim(-0.2,0.1)
plt.xlabel(r'log([NII]/H$\mathrm{\alpha}$) (Palomar)',fontsize=16)
plt.ylabel(r'log([NII]/H$\mathrm{\alpha}$) (HST)',fontsize=16)
plt.tick_params(axis='both',direction='in',labelsize=14)

# Histogram on the x margin
ax2_margx = fig2.add_subplot(gs[0,0:3])
plt.hist(NII_Halpha,histtype='stepfilled',color='silver',bins=10)
plt.hist(NII_Halpha,histtype='step',color='k',bins=10)
plt.ylim(0,4)
plt.setp(ax2_margx.get_xticklabels(), visible=False)
for i in range(len(ax_margx.get_yticklabels())-1):
    plt.setp(ax2_margx.get_yticklabels()[i], visible=False)
plt.setp(ax2_margx.yaxis.get_minorticklines(), visible=False)
plt.tick_params(axis='both',direction='in',labelsize=12)
plt.xlim(-0.6,0.3)

# Histogram on the y margin 
ax2_margy = fig2.add_subplot(gs[1:4,-1])
plt.hist(NII_Halpha_HST,histtype='stepfilled',color='silver',orientation='horizontal',bins=10)
plt.hist(NII_Halpha_HST,histtype='step',color='k',orientation='horizontal',bins=10)
plt.xlim(0,4)
plt.setp(ax2_margy.get_yticklabels(), visible=False)
for i in range(len(ax_margy.get_xticklabels())-1):
    plt.setp(ax2_margy.get_xticklabels()[i], visible=False)
plt.setp(ax2_margy.xaxis.get_minorticklines(), visible=False)
plt.ylim(-0.2,0.1)
plt.locator_params(axis='y', nbins=5)
plt.tick_params(axis='both',direction='in',labelsize=12)

plt.subplots_adjust(wspace=0.,hspace=0.0)

#plt.savefig('/mnt/data/lhermosa/HLA_data/line_ratios/NII_Halpha_HSTvsPalomar_labels.png',bbox_inches='tight',pad_inches=0.2)
#plt.savefig('/mnt/data/lhermosa/HLA_data/line_ratios/NII_Halpha_HSTvsPalomar_labels.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)



'''
fig2 = plt.figure(figsize=(7,6))
gs2 = GridSpec(4,4)
ax21 = fig2.add_subplot(gs[1:4,0:3])
plt.plot(SII_Halpha,SII_Halpha_HST-SII_Halpha,'ko',linestyle='None')
#plt.fill_between(np.linspace(-0.6,0.6),np.linspace(-0.6,0.6)+0.2,np.linspace(-0.6,0.6)-0.2,color='paleturquoise',alpha=0.5)
#plt.fill_between(np.linspace(-0.6,0.6),np.linspace(-0.6,0.6)+0.1,np.linspace(-0.6,0.6)-0.1,color='lightsalmon',alpha=0.5)
plt.plot(np.linspace(-0.6,0.6),np.zeros(len(np.linspace(-0.6,0.6))),c='grey',linestyle='--')
# To include the labels of each galaxy in the plot...
for i in range(len(SII_Halpha)):
    plt.text(SII_Halpha[i]-0.04,SII_Halpha_HST[i]-SII_Halpha[i]+0.02,gal[i],fontsize=8)
plt.locator_params(axis='y', nbins=5)
plt.xlim(-0.6,0.3)
#plt.ylim(-0.2,0.1)
plt.xlabel(r'log([NII]/H$\mathrm{\alpha}$) (Palomar)',fontsize=16)
plt.ylabel(r'log([NII]/H$\mathrm{\alpha}$) (HST)',fontsize=16)
plt.tick_params(axis='both',direction='in',labelsize=14)

# Histogram on the x margin
ax2_margx = fig2.add_subplot(gs[0,0:3])
plt.hist(SII_Halpha,histtype='stepfilled',color='silver',bins=10)
plt.hist(SII_Halpha,histtype='step',color='k',bins=10)
plt.ylim(0,4)
plt.setp(ax2_margx.get_xticklabels(), visible=False)
plt.setp(ax2_margx.get_yticklabels()[0], visible=False)
plt.setp(ax2_margx.get_yticklabels()[1], visible=False)
plt.setp(ax2_margx.yaxis.get_minorticklines(), visible=False)
plt.tick_params(axis='both',direction='in',labelsize=12)
plt.xlim(-0.6,0.3)

# Histogram on the y margin 
ax2_margy = fig2.add_subplot(gs[1:4,-1])
plt.hist(SII_Halpha_HST-SII_Halpha,histtype='stepfilled',color='silver',orientation='horizontal',bins=10)
plt.hist(SII_Halpha_HST-SII_Halpha,histtype='step',color='k',orientation='horizontal',bins=10)
plt.xlim(0,4)
plt.setp(ax2_margy.get_yticklabels(), visible=False)
plt.setp(ax2_margy.get_xticklabels()[0], visible=False)
plt.setp(ax2_margy.get_xticklabels()[1], visible=False)
plt.setp(ax2_margy.xaxis.get_minorticklines(), visible=False)
#plt.ylim(-0.2,0.1)
plt.locator_params(axis='y', nbins=5)
plt.tick_params(axis='both',direction='in',labelsize=12)

plt.subplots_adjust(wspace=0.,hspace=0.0)
'''
