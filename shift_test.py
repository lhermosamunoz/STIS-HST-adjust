'''
Import to shift the individual spectra and then combine them individually
'''
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from pyraf import iraf
from astropy.io import fits


####### Define PATHS #######
path      = '/mnt/data/lhermosa/HLA_data/'
targname  = 'NGC5055'					# From the header. Change this for each case!
galname   = 'NGC5055'					# Change this for each case!

####### Define FUNCTIONS ######
def getHeaderVar(filePath,*varList):
	#get variables from FITS header
	fileFits = fits.open(filePath)
	i = np.array([])
	for index in varList:
		fitsVar = fileFits[0].header[str(index)]
		i = np.append(i,fitsVar)
	fileFits.close()
	return i

def getHeaderVar1(filePath,*varList):
	#get variables from FITS header
	fileFits = fits.open(filePath)
	i = np.array([])
	for index in varList:
		fitsVar = fileFits[1].header[str(index)]
		i = np.append(i,fitsVar)
	fileFits.close()
	return i

################################################################################################################################
############################################################# MAIN #############################################################
################################################################################################################################

RA_aper, DEC_aper, PA_aper = np.array([]), np.array([]), np.array([])
platesc, exptime           = np.array([]), np.array([])
minwave,filename           = np.array([]), np.array([])

for i in np.sort(glob(path+'NGC*/shifts/*x2d.fits')):
	var = getHeaderVar(i,'TARGNAME')		# Select the galaxy and get the data from its header
	if var[0]==targname:
		platesc  = np.append(platesc,getHeaderVar(i,'PLATESC')/3600.)	# degree/pix
		exptime  = np.append(exptime,getHeaderVar(i,'TEXPTIME'))
		minwave  = np.append(minwave,getHeaderVar(i,'MINWAVE'))
		filename = np.append(filename,getHeaderVar(i,'FILENAME'))
		RA_aper  = np.append(RA_aper,getHeaderVar1(i,'RA_APER'))
		DEC_aper = np.append(DEC_aper,getHeaderVar1(i,'DEC_APER'))
		PA_aper  = np.append(PA_aper,getHeaderVar1(i,'PA_APER'))

#
# This is for dividing each image depending on the position angle, and store their position in the arrays for calculating the shifts.
refpa   = PA_aper[0]
paaper  = np.copy(PA_aper)
ixfinal = []
n = 0

while n < 10:
    # Now we identify the spectrums with/out the same position angle
    ix   = np.where(paaper == refpa)
    noix = np.where(paaper != refpa)
    ixfinal.append(ix[0].tolist())
    # Now we redefine the refpa and identify again
    if np.size(noix) == 0:
	print('Only one position angle for all!')
	break
    elif np.size(ix) == 0:
	print('Something went wrong! Not finding any more position angles.')
	break
    else:
	# If the aperture angle has been set to zero, then look for the next one different from 0 and use it as the new refpa
	# If the aperture angle is directly not zero, then refpa will be directly PA_aper of noix[0]
	if paaper[noix[0][0]] == 0:
	    index = np.where(paaper[noix[0]] != 0)[0]
	    if np.size(index) == 0: 				# This means that all the angles have been set to zero and we can continue
		print('All angles already distributed!')
		print('')
		print('The final distribution of PA is: '+str(ixfinal))
		break
	    refpa = paaper[noix[0][index]][0]
	    print('Found paaper == 0, so new refpa is '+str(refpa))
	else:
	    refpa  = paaper[noix[0][0]]
	    print('Found paaper != 0, new refpa is '+str(refpa))
	paaper[ix] = 0	# Remove the position angles already analized from the vector
    n+=1
    print('Iteration '+str(n)+' done!')

#
# Now we have to calculate the shifts
#
# The position RA/DEC of the first image is going to be the reference for making the shift. Then we have to calculate the diference in the position of the rest of the images with respect to this one. 
# Create a list with the shifts in arcsecs and then use the plate scale to transform the shift into pixels. This will be used as the input for the IMSHIFT task in IRAF.

# n-elements in ixfinal and m-elements in the n-elements
# only those which minwave is around 6500 A (6000<minwave<7000)
# calculate the shift in RA and DEC being the first element the reference
# save the shifts in a file
# 
shift_fin   = []
shift_temp  = []
for id1 in range(len(ixfinal)):
    RA_temp    = RA_aper[ixfinal[id1]]
    DEC_temp   = DEC_aper[ixfinal[id1]]
    PA_temp    = PA_aper[ixfinal[id1]]
    lam_temp   = minwave[ixfinal[id1]]
    pltsc_temp = platesc[ixfinal[id1]]
    exp_temp   = minwave[ixfinal[id1]]
    name_temp  = filename[ixfinal[id1]]
    for id2 in range(0,len(RA_temp)):
    	if lam_temp[id2] < 7000. and lam_temp[id2] >6000.:
	    refra  = RA_temp[0]
	    refdec = DEC_temp[0]
	    refpa  = PA_temp[0]
	    shift_ra  = RA_temp[id2]-refra
	    shift_dec = DEC_temp[id2]-refdec
	    shift_temp.append([name_temp[id2],shift_ra/pltsc_temp[id2],shift_dec/pltsc_temp[id2],refpa])
	    #f = open(path+galname+'/shift_pix_'+targname+'_n'+str(id1)+'.txt','a')
	    #f.write('\n'+str(shift_ra)+' '+str(shift_dec)+' '+str(refpa)+' '+str(name_temp[id2]))
	    #f.close()
	else: 
	    print('The element '+str(ixfinal[id1][id2])+' has been excluded because it has a min wavelength larger or smaller than 6000.-7000. angstroms')
	    print('In the vector corresponding to these images, the PA will be replaced by 0.')
	    shift_temp.append([name_temp[id2],0.,0.,0.])
	    #f = open(path+galname+'/shift_pix_'+targname+'_lam'+str(lam_temp[id2])+'_n'+str(id1)+'.txt','a')
	    #f.write('\n'+str(0.)+' '+str(0.)+' '+str(refpa)+' '+str(name_temp[id2]))
	    #f.close()
    shift_fin.append(shift_temp)
    shift_temp = []


##########################################################################################################################################################
######################################################### IMSHIFT + IMCOMBINE ############################################################################
##########################################################################################################################################################
# 
# Perform the shifts with iraf/pyraf
# We need the imshift task: imshift input output xshift yshift (or file with shifts in cols 1 & 2)
# The initial list will be *x2d.fits[1] and the output list will be *shift.fits
# 
list_temp  = []
list_total = []
iraf.cd(path+galname+'/shifts')
for index in range(len(shift_fin)):
    arr1 = shift_fin[index]
    for index2 in range(len(arr1)):
	i_nam = arr1[index2][0]         # Initial name for imshift
	xs    = arr1[index2][1]
	ys    = arr1[index2][2]
	posa  = arr1[index2][3]
	list_in  = i_nam.replace('.fits','.fits[1,overwrite+]')
	list_ot  = list_in.replace('x2d','shift')
	list_out = list_ot.replace('.fits[1,overwrite+]','.fits')
	list_total.append(list_temp)
	if not os.path.exists(list_out):
	    iraf.imshift(input=list_in,output=list_out,xshift=xs,yshift=ys)
	    print('File '+str(i_nam)+' changed to '+str(list_out)+'!')





iraf.cd(path+'scripts')
