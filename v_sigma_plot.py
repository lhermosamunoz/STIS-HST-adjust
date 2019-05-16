import numpy as np
import matplotlib.pyplot as plt

# We want to do a figure representing the velocity dispersion vs the velocity of the individual components
# We will do two separate subplots for each of the individual components
# To do this, we define the plot zone apart from the code 
def lines(comp):
     plt.hlines(0.,0.,1000.,'k')
     plt.vlines(400,-600.,600.,'k')
     plt.hlines(100,300,400,'k')
     plt.hlines(-100,300,400,'k')
     plt.vlines(300,100,600,'k')
     plt.vlines(300,-600,-100,'k')
     plt.xlim(0,1000)
     plt.ylim(-600,600)
     plt.fill_between(np.arange(400.,1001,5.),0.,600.,color='yellow',alpha=0.3)
     plt.fill_between(np.arange(400.,1001,5.),-600.,0.,color='blue',alpha=0.2)
     plt.fill_between(np.arange(300,401.,5.),100,600.,color='orange',alpha=0.4)
     plt.fill_between(np.arange(300,401.,5.),-600,-100.,color='orange',alpha=0.4)
     
     plt.text(30,-580,'Rotation',fontsize=14,weight='light')
     plt.text(300,550,'Candidates',fontsize=14,rotation=70,weight='light')
     plt.text(840,30,'Inflows',fontsize=14,weight='light')
     plt.text(830,-50,'Outflows',fontsize=14,weight='light')
     plt.text(560,510,comp,fontsize=18,weight='normal')
     
     plt.xticks(fontsize=16)
     plt.yticks(fontsize=16)

############################################################################################################
# Data
v_narrow = np.array([80.,134.,-275.,395.,-103.,70.,527.,55.,127.])
v_secondary = np.array([362.,-101.,501.,66.])
sig_narrow = np.array([63.,172.,146.,198.,157.,108.,198.,90.,83.])
sig_secondary = np.array([281.,655.,249.,554.])
erv_N = np.array([1.,29.,0.,4.,47.,15.,40.,19.,12.])
erv_S = np.array([0.,378.,38.,152.])
ersig_N = np.array([19.,12.,14.,22.,18.,25.,17.,11.,19.])
ersig_S = np.array([24.,18.,50.,12.])

############################################################################################################
# Define the main plot with two subplots     
fig = plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
lines('Narrow component')
plt.errorbar(sig_narrow,v_narrow,erv_N,ersig_N,'ro',markersize=9,capsize=5)
plt.xlabel('Velocity dispersion (km s$^{-1}$)',fontsize=15)
plt.ylabel('Velocity (km s$^{-1}$)',fontsize=15)

# --------------------------------------------------------
plt.subplot(1,2,2)
lines('Second component')
plt.errorbar(sig_secondary,v_secondary,erv_S,ersig_S,'ro',markersize=9,capsize=5)
plt.xlabel('Velocity dispersion (km $s^{-1}$)',fontsize=15)

plt.savefig('/mnt/data/lhermosa/HLA_data/v_vs_sigma.png',bbox_inches='tight',pad_inches=0.2)
plt.savefig('/mnt/data/lhermosa/HLA_data/v_vs_sigma.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)
