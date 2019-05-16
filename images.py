'''
Script to load and cut the images of the galaxies to the desired scale
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from pyraf import iraf
import folders
import os

# Define the path to the data
'''
#basicPath = '/media/laura/UUI/PhD/HLA_data/'
#basicPath = '/media/lhermosa/UUI/PhD/HLA_data/'
basicPath = '/mnt/data/lhermosa/HLA_data/'
gal = input('Which galaxy inside the folder HLA_data?: ')
path = basicPath+gal+'/Images/'
print(os.listdir(path))
file_name = input('Which file inside the previous path?: ')
'''
filePath = folders.path_gal
basicPath = folders.parentFold
galaxy = folders.gal
angles = np.loadtxt(basicPath+'gal_angles.txt')
PA = angles[:,0]
PA_majAxis = angles[:,1]
slit_width = angles[:,2]
parsecs = angles[:,3]
# Read the image and obtain the info from the header. 
# In this case we are using images that are already full reduced, so there is no need of combining several exposures
for i in range(len(filePath)):
   gal = galaxy[i]
   fileName = filePath[i]
   path = basicPath+gal+folders.imaFold

   # Open the image
   hdulist = fits.open(fileName)
   hdulist.info()

   # Normally the data is stored inside the first layer of the .fits file
   # but change it if different in a particular case!!!!
   header = hdulist[0].header
   hdu = hdulist[1]
   image_data = hdu.data

   hdulist.close()

   # Extract the central coordinates and transform them into hhmmss ddmmss
   # In this way we can crop the images using the true coords and sizes
   ra_deg = header['RA_TARG']
   dec_deg = header['DEC_TARG']
   c = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
   coords = SkyCoord.from_name(gal)
   print('The central coordinates of the pointing are '+str(c.ra.hms)+' '+str(c.dec.dms))
   print('The central coordinates of the galaxy are '+str(coords.ra.hms)+' '+str(coords.dec.dms))
   print('')

   # Now we plot it to show how it looks like in a logarithmic scale. 
   # In order to make it easier to visualize something, it is better 
   # to do an histogram and set the more appropiate limits to the plot
   plt.ion()
   plt.show()
   plt.imshow(image_data,cmap='gray',norm=LogNorm())
   if not os.path.exists(path+'central_coordinates.txt'):
      print('Please, look for the coordinates in the images and put here the correspondent value in pixels! (imexam task, press w + A in ds9!) ')
      iraf.cd(path)
      print('Moving to the directory... '+os.getcwd())
      iraf.set(stdimage = 'imt4096')
      iraf.imexam(file_name+'[1]',frame='1',logfile='central_coordinates.txt',wcs='image')
      iraf.cd(basicPath+'scripts/')
      print('Moving back to the directory... '+os.getcwd())

   if not os.path.exists(path+'example_cutout.fits'):
      true_center = np.genfromtxt(path+'central_coordinates.txt')
      xpos = true_center[0] #input('What is the position of the nucleus in pixels X?: ')
      ypos = true_center[1] #input('What is the position of the nucleus in pixels Y?: ')
      ra_pos = true_center[2]
      dec_pos = true_center[3]
      true_coords = SkyCoord(ra=ra_pos*u.degree, dec=dec_pos*u.degree, frame='fk5')
      wcs = WCS(hdu.header)
      wcs.wcs.crval = [true_coords.ra.deg,true_coords.dec.deg]
      wcs.wcs.crpix = [xpos, ypos]
      '''
      # Only if the info of the header is not available
      #rho = np.pi/3.
      #scale = 0.05/3600.  # Plate scale!
      #wcs.wcs.cd = [[scale*np.cos(rho), -scale*np.sin(rho)],
      #              [scale*np.sin(rho), scale*np.cos(rho)]]
      '''
      # Cut the image with all the previous information. The angular scale is the same as the MEGARA FoV
      cutout = Cutout2D(image_data, (float(xpos),float(ypos)), u.Quantity((12,12),u.arcsec), wcs=wcs) # data, position of the center, angular size to cut, wcs 
    
      # Create a new image .fits with the new cut applied
      hdu.data = cutout.data
      # Update the FITS header with the cutout WCS
      hdu.header.update(cutout.wcs.to_header())
      # Write the cutout to a new FITS file
      cutout_filename = 'example_cutout.fits'
      hdu.writeto(path+cutout_filename, overwrite=True)
    
      ##############################  PLOT  ###############################################
      plt.close()
      plt.imshow(cutout.data,cmap='gray',norm=LogNorm())
      plt.gca().invert_yaxis()
      plt.savefig(path+'12x12_selection.pdf',format='pdf')
      plt.savefig(path+'12x12_selection.png')
      # Print in the original image the selected section to see that everything is working!
      plt.figure()
      plt.imshow(image_data,cmap='gray',norm=LogNorm())
      cutout.plot_on_original()
      plt.gca().invert_yaxis()
      plt.savefig(path+'full_selection.pdf',format='pdf')
      plt.savefig(path+'full_selection.png')

   
   #######################################################################################################
   #######################################################################################################
   #			THIS IS FOR CREATING THE SHARP-DIVIDED IMAGES
   #######################################################################################################
   #######################################################################################################
   if not os.path.exists(path+'sharp_divided.fits'):
      from pyraf import iraf
      # Run this in a pyraf terminal
      iraf.imdelete(path+'filtered_cutout.fits',verify='No')
      iraf.imdelete(path+'sharp_divided.fits',verify='No')
      iraf.median(path+'example_cutout.fits[1]',path+'filtered_cutout.fits',xwindow=30,ywindow=30)
      iraf.imarith(operand1=path+'example_cutout.fits[1]',op='/',operand2=path+'filtered_cutout.fits',result=path+'sharp_divided.fits')

   ############################# FINAL PLOT OF THE SHARP DIVIDED ########################################
   # First we have to load the new image and header to have the new coordinates
   hdulist1 = fits.open(path+'sharp_divided.fits')
   sharp_data = hdulist1[0].data
   sharp_header = hdulist1[0].header
   hdulist1.close()
   x0 = sharp_header['CRPIX1']
   y0 = sharp_header['CRPIX2']
   RA0 = sharp_header['CRVAL1']
   DEC0 = sharp_header['CRVAL2']
   # Information for the spatial scale
   x_text = np.arange(3.,4.1,0.1)
   spat_scale = parsecs[i] # input('What is the spatial scale of the galaxy? (in pc/arcsec): ')
   # Calculate the shape of the slit with the corresponding angle and starting point
   x1 = 0 #float(input('What is the 0 point of the slit in the spectra? (RA_APER): '))-RA0
   y1 = 0 #float(input('What is the 0 point of the slit in the spectra? (DEC_APER): '))-DEC0
   angle = PA[i] + 90 		# float(input('What is the angle of the slit in the spectra? (PA_APER): '))+90
   angle2 = PA_majAxis[i] + 90 	# float(input('What is the angle of the major axis of the galaxy? (deg): '))+90
   pltscl = slit_width[i] 	# float(input('Slit width? (0.1/0.2 arcsec?): '))
   x2 = x1 + np.cos((np.pi/180.)*angle)*11
   y2 = y1 + np.sin((np.pi/180.)*angle)*11
   x3 = np.cos((np.pi/180.)*angle2)*11
   y3 = np.sin((np.pi/180.)*angle2)*11

   plt.close('all')
   fig = plt.figure(figsize=(9,9))
   ax = fig.add_subplot(111)
   plt.imshow(sharp_data,cmap='jet',norm=LogNorm(),origin=[x1,y1],extent=[-5,5,-5,5])
   plt.plot(x1,y1,'k+',markersize=8)
   plt.plot(x_text,3.8+np.zeros(len(x_text)),'w-',linewidth=2)
   plt.text(3.,4.,str(int(spat_scale))+' pc',color='white',fontsize=20)
   plt.ylim(-5,5)
   plt.xlim(-5,5)
 
   for tick in ax.get_xticklines():
      tick.set_color('white')
      tick.set_markersize(12)
      tick.set_markeredgewidth(3)
   for tick in ax.get_xticklabels():
      tick.set_color('black')
   for minortick in ax.xaxis.get_minorticklines():
      minortick.set_color('white')
   for tick in ax.get_yticklines():
      tick.set_color('white')
      tick.set_markersize(12)
      tick.set_markeredgewidth(3)
   for tick in ax.get_yticklabels():
      tick.set_color('black')
     
   plt.xlabel(r'$\Delta \alpha$ [arcsec]',fontsize=18)
   plt.ylabel(r'$\Delta \delta$ [arcsec]',fontsize=18)
   plt.tick_params(axis='both', labelsize=20)
   if not os.path.exists(path+'final_sharp_divided_'+gal+'.png'):
      plt.savefig(path+'final_sharp_divided_'+gal+'.png')
      plt.savefig(path+'final_sharp_divided_'+gal+'.pdf',format='pdf')

   #plt.plot([-x2,x2],[-y2,y2],'w-')
   plt.plot([-x2-pltscl/2.,x2-pltscl/2.],[-y2-pltscl/2.,y2-pltscl/2.],'w-',linewidth=2)
   plt.plot([-x2+pltscl/2.,x2+pltscl/2.],[-y2+pltscl/2.,y2+pltscl/2.],'w-',linewidth=2)
   if not os.path.exists(path+'final_slit_sharp_divided_'+gal+'.png'):
      plt.savefig(path+'final_slit_sharp_divided_'+gal+'.png',bbox_inches='tight',pad_inches=0.2)
      plt.savefig(path+'final_slit_sharp_divided_'+gal+'.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)
 
   plt.plot([-x3,x3],[-y3,y3],'w--')
   plt.plot(0,0,'k+')
   if angle2 != 90:
      if not os.path.exists(path+gal+'_image.png'):
         plt.savefig(path+gal+'_image.png',bbox_inches='tight',pad_inches=0.2)
         plt.savefig(path+gal+'_image.pdf',format='pdf',bbox_inches='tight',pad_inches=0.2)
   
   print('Done folder '+gal)
