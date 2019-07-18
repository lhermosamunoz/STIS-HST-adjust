import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def BPTs_alltogether(NII_Halpha,SII_Halpha,OI_Halpha,OIII_Halpha):
   '''
   Function to plot BPTs. 
   The parameters needed are the NII/Halpha, SII/Halpha, OI/Halpha and OIII/Halpha ratios
   It returns a plot with the three BPTs all together. 
   '''
   plt.figure(figsize=(21,7))
   gs = gridspec.GridSpec(1,3)
   ax0 = plt.subplot(gs[0])
   ax0.tick_params(direction='in',which='both',top=True,right=True)
   plt.ylim(-1.0,2.0)
   plt.xlim(-2.0,1.1)
   plt.xlabel(r'log([NII]/H$\alpha$)',fontsize=18)
   plt.ylabel(r'log([OIII]/H$\beta$)',fontsize=18)
   plt.text(-1.5,-0.7,'HII-SF',fontsize=16)
   plt.text(0.5,-0.6,'LINER',fontsize=16)
   plt.text(-0.5,1.6,'Seyfert',fontsize=16)
   plt.tick_params(axis='both', labelsize=16)
   x_SeyLin = np.linspace(-2.0,0.04)
   x_AGNSF = np.linspace(-2.0,0.24)
   y_SeyLin = (0.61/(x_SeyLin-0.05)) + 1.30
   y_AGNSF = (0.61/(x_AGNSF-0.47)) + 1.19
   plt.plot(x_SeyLin,y_SeyLin,'k-')
   plt.plot(x_AGNSF,y_AGNSF,'k-')
   plt.plot(NII_Halpha,OIII_Halpha,'bo')

   ax1 = plt.subplot(gs[1],sharey=ax0)
   ax1.tick_params(direction='in',which='both',top=True,right=True)
   plt.setp(ax1.get_yticklabels(),visible=False)
   plt.ylim(-1.0,2.0)
   plt.xlim(-1.95,1.1)
   plt.xlabel(r'log([SII]/H$\alpha$)',fontsize=18)
   plt.text(-1.5,-0.7,'HII-SF',fontsize=16)
   plt.text(0.5,-0.6,'LINER',fontsize=16)
   plt.text(-0.5,1.6,'Seyfert',fontsize=16)
   plt.tick_params(axis='x', labelsize=16)
   x_SeyLin = np.linspace(-0.31,1.0)
   x_AGNSF = np.linspace(-2.0,0.04)
   y_SeyLin = 1.89 * x_SeyLin + 0.76
   y_AGNSF = (0.72/(x_AGNSF-0.32)) + 1.30
   plt.plot(x_SeyLin,y_SeyLin,'k-')
   plt.plot(x_AGNSF,y_AGNSF,'k-')
   plt.plot(SII_Halpha,OIII_Halpha,'bo')

   ax2 = plt.subplot(gs[2],sharey=ax0)
   ax2.tick_params(direction='in',which='both',top=True,right=True)
   plt.setp(ax2.get_yticklabels(),visible=False)
   plt.ylim(-1.0,2.0)
   plt.xlim(-1.98,0.7)
   plt.xlabel(r'log([OI]/H$\alpha$)',fontsize=18)
   plt.text(-1.5,-0.7,'HII-SF',fontsize=16)
   plt.text(0.2,-0.6,'LINER',fontsize=16)
   plt.text(-0.5,1.6,'Seyfert',fontsize=16)
   plt.tick_params(axis='x', labelsize=16)
   x_SeyLin = np.linspace(-1.11,0.6)
   x_AGNSF = np.linspace(-2.0,-0.7)
   y_SeyLin = 1.18 * x_SeyLin + 1.30
   y_AGNSF = (0.73/(x_AGNSF+0.59)) + 1.33
   plt.plot(x_SeyLin,y_SeyLin,'k-')
   plt.plot(x_AGNSF,y_AGNSF,'k-')
   plt.plot(OI_Halpha,OIII_Halpha,'bo')
   plt.subplots_adjust(hspace=.0,wspace=.02)

def BPT_NII(NII_Halpha,OIII_Halpha):
   '''
   Function to plot the NII/Halpha vs OIII/Halpha BPTs. 
   The parameters needed are the NII/Halpha and OIII/Halpha ratios
   '''
   plt.figure(figsize=(8,8))
   plt.ylim(-1.0,2.0)
   plt.xlim(-2.0,1.0)
   plt.xlabel(r'log([NII]/H$\alpha$)',fontsize=18)
   plt.ylabel(r'log([OIII]/H$\beta$)',fontsize=18)
   plt.text(-1.5,-0.7,'HII-SF',fontsize=16)
   plt.text(0.5,-0.6,'LINER',fontsize=16)
   plt.text(-0.5,1.6,'Seyfert',fontsize=16)
   plt.tick_params(axis='both', labelsize=16)
   x_SeyLin = np.linspace(-2.0,0.04)
   x_AGNSF = np.linspace(-2.0,0.24)
   y_SeyLin = (0.61/(x_SeyLin-0.05)) + 1.30
   y_AGNSF = (0.61/(x_AGNSF-0.47)) + 1.19
   plt.plot(x_SeyLin,y_SeyLin,'k-')
   plt.plot(x_AGNSF,y_AGNSF,'k-')
   # add the data ....
   plt.plot(NII_Halpha,OIII_Halpha,'bo')
   # plt.errorbar(NII_Halpha,OIII_Halpha,errorOIII,errorNII,'bo')

def BPT_OI(OI_Halpha,OIII_Halpha):
   '''
   Function to plot the OI/Halpha vs OIII/Halpha BPTs. 
   The parameters needed are the OI/Halpha and OIII/Halpha ratios
   '''
   plt.figure(figsize=(8,8))
   plt.ylim(-1.0,2.0)
   plt.xlim(-2.0,0.7)
   plt.xlabel(r'log([OI]/H$\alpha$)',fontsize=18)
   plt.ylabel(r'log([OIII]/H$\beta$)',fontsize=18)
   plt.text(-1.5,-0.7,'HII-SF',fontsize=16)
   plt.text(0.2,-0.6,'LINER',fontsize=16)
   plt.text(-0.5,1.6,'Seyfert',fontsize=16)
   plt.tick_params(axis='both', labelsize=16)
   x_SeyLin = np.linspace(-1.11,0.6)
   x_AGNSF = np.linspace(-2.0,-0.7)
   y_SeyLin = 1.18 * x_SeyLin + 1.30
   y_AGNSF = (0.73/(x_AGNSF+0.59)) + 1.33
   plt.plot(x_SeyLin,y_SeyLin,'k-')
   plt.plot(x_AGNSF,y_AGNSF,'k-')
   # add the data ....
   plt.plot(OI_Halpha,OIII_Halpha,'bo')
   # plt.errorbar(OI_Halpha,OIII_Halpha,errorOIII,errorOI,'bo')

def BPT_SII(SII_Halpha,OIII_Halpha):
   '''
   Function to plot the SII/Halpha vs OIII/Halpha BPTs. 
   The parameters needed are the SII/Halpha and OIII/Halpha ratios
   '''
   plt.figure(figsize=(8,8))
   plt.ylim(-1.0,2.0)
   plt.xlim(-2.0,1.0)
   plt.xlabel(r'log([SII]/H$\alpha$)',fontsize=18)
   plt.ylabel(r'log([OIII]/H$\beta$)',fontsize=18)
   plt.text(-1.5,-0.7,'HII-SF',fontsize=16)
   plt.text(0.5,-0.6,'LINER',fontsize=16)
   plt.text(-0.5,1.6,'Seyfert',fontsize=16)
   plt.tick_params(axis='both', labelsize=16)
   x_SeyLin = np.linspace(-0.31,1.0)
   x_AGNSF = np.linspace(-2.0,0.04)
   y_SeyLin = 1.89 * x_SeyLin + 0.76
   y_AGNSF = (0.72/(x_AGNSF-0.32)) + 1.30
   plt.plot(x_SeyLin,y_SeyLin,'k-')
   plt.plot(x_AGNSF,y_AGNSF,'k-')
   # add the data ....
   plt.plot(SII_Halpha,OIII_Halpha,'bo')
   # plt.errorbar(SII_Halpha,OIII_Halpha,errorOIII,errorSII,'bo')

