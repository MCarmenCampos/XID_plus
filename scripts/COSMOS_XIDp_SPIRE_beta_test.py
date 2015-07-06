import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import XIDp_mod_beta as xid_mod

#Folder containing maps
imfolder='/research/astrodata/fir/hermes/xid/XID_2014/SMAP_images_v4.2/'
#field
field='COSMOS'
#SMAP version
SMAPv='4.2'


# In[7]:

pswfits=imfolder+field+'_image_250_SMAP_v'+SMAPv+'.fits'#SPIRE 250 map
pmwfits=imfolder+field+'_image_350_SMAP_v'+SMAPv+'.fits'#SPIRE 350 map
plwfits=imfolder+field+'_image_500_SMAP_v'+SMAPv+'.fits'#SPIRE 500 map


# In[8]:

#Folder containing prior input catalogue
folder="/research/astrodata/fir/hermes/www/xid/rel0712/"
#prior catalogue
prior_cat="mod_cosmos-xid-pepprior-0512.fits.gz"
hdulist = fits.open(folder+prior_cat)
fcat=hdulist[1].data
hdulist.close()
inra=fcat['INRA']
indec=fcat['INDEC']
f_src=fcat['F250']
df_src=f_src
nrealcat=fcat.size
bkg250=fcat['bkg250'][0]
bkg350=fcat['bkg350'][0]
bkg500=fcat['bkg500'][0]


# Open images and noise maps and use WCS module in astropy to get header information

# In[9]:

#-----250-------------
hdulist = fits.open(pswfits)
im250phdu=hdulist[0].header
im250=hdulist[1].data*1.0E3
nim250=hdulist[2].data*1.0E3
w_250 = wcs.WCS(hdulist[1].header)
pixsize250=3600.0*w_250.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----350-------------
hdulist = fits.open(pmwfits)
im350phdu=hdulist[0].header
im350=hdulist[1].data*1.0E3
nim350=hdulist[2].data*1.0E3
w_350 = wcs.WCS(hdulist[1].header)
pixsize350=3600.0*w_350.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----500-------------
hdulist = fits.open(plwfits)
im500phdu=hdulist[0].header
im500=hdulist[1].data*1.0E3
nim500=hdulist[2].data*1.0E3
w_500 = wcs.WCS(hdulist[1].header)
pixsize500=3600.0*w_500.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()

# Since I am only testing, redo this so that I only fit sources within a given range of the mean ra and dec position of the prior list

# In[12]:

#define range
ra_mean=np.mean(inra)
dec_mean=np.mean(indec)
p_range=0.08
#check if sources are within range and if the nearest pixel has a finite value 

sgood=(inra > ra_mean-p_range) & (inra < ra_mean+p_range) & (indec > dec_mean-p_range) & (indec < dec_mean+p_range)
inra=inra[sgood]
indec=indec[sgood]
n_src=sgood.sum()
print 'fitting '+str(n_src)+' sources'



# Point response information, at the moment its 2D Gaussian,

#pixsize array (size of pixels in arcseconds)
pixsize=np.array([pixsize250,pixsize350,pixsize500])
#point response function for the three bands
prfsize=np.array([18.15,25.15,36.3])
#use Gaussian2DKernel to create prf (requires stddev rather than fwhm hence pfwhm/2.355)
from astropy.convolution import Gaussian2DKernel



prior250=xid_mod.prior(im250,nim250,w_250,im250phdu)
prior250.prior_cat(inra,indec,prior_cat)
prior250.prior_bkg(bkg250,2)
prior350=xid_mod.prior(im350,nim350,w_350,im350phdu)
prior350.prior_cat(inra,indec,prior_cat)
prior350.prior_bkg(bkg350,2)
prior500=xid_mod.prior(im500,nim500,w_500,im500phdu)
prior500.prior_cat(inra,indec,prior_cat)
prior500.prior_bkg(bkg500,2)

#thdulist,prior250,prior350,prior500,posterior=xid_mod.fit_SPIRE(prior250,prior350,prior500)

#-----------fit using real beam--------------------------
#PSF_250,px_250,py_250=xid_mod.SPIRE_PSF('../hsc-calibration/0x5000241aL_PSW_bgmod9_1arcsec.fits',pixsize[0])
#PSF_350,px_350,py_350=xid_mod.SPIRE_PSF('../hsc-calibration/0x5000241aL_PMW_bgmod9_1arcsec.fits',pixsize[1])
#PSF_500,px_500,py_500=xid_mod.SPIRE_PSF('../hsc-calibration/0x5000241aL_PLW_bgmod9_1arcsec.fits',pixsize[2])
##---------fit using Gaussian beam-----------------------
prf250=Gaussian2DKernel(prfsize[0]/2.355,x_size=101,y_size=101)
prf250.normalize(mode='peak')
prf350=Gaussian2DKernel(prfsize[1]/2.355,x_size=101,y_size=101)
prf350.normalize(mode='peak')
prf500=Gaussian2DKernel(prfsize[2]/2.355,x_size=101,y_size=101)
prf500.normalize(mode='peak')

pind250=np.arange(0,101,1)*1.0/pixsize[0] #get 250 scale in terms of pixel scale of map
pind350=np.arange(0,101,1)*1.0/pixsize[1] #get 350 scale in terms of pixel scale of map
pind500=np.arange(0,101,1)*1.0/pixsize[2] #get 500 scale in terms of pixel scale of map

prior250.set_prf(prf250.array,pind250,pind250)
prior350.set_prf(prf350.array,pind350,pind350)
prior500.set_prf(prf500.array,pind500,pind500)

prior250.get_pointing_matrix()
prior350.get_pointing_matrix()
prior500.get_pointing_matrix()

fit_data,chains,iter=xid_mod.lstdrv_SPIRE_stan(prior250,prior350,prior500)
posterior=xid_mod.posterior_stan(fit_data[:,:,0:-1],prior250.nsrc)
thdulist=xid_mod.create_XIDp_SPIREcat(posterior,prior250,prior350,prior500)
#----------------------------------------------------------


output_folder='/research/astro/fir/HELP/XID_plus_output/'
thdulist.writeto(output_folder+'XIDp_SPIRE_beta_'+field+'_dat_small_0.08_Gauss.fits')
outfile=output_folder+'XIDp_SPIRE_beta_test_small_0.08_Gauss.pkl'
with open(outfile, 'wb') as f:
    pickle.dump({'psw':prior250,'pmw':prior350,'plw':prior500,'posterior':posterior},f)
