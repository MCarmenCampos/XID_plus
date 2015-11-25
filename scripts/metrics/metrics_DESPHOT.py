#---import modules---
from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use('PDF')
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

#import XIDp_mod_beta
import pickle
pdf_pages=PdfPages("error_density_flux_test_DESPHOT.pdf")

from metrics_module import *
#---Read in DESPHOT catalogue---

#folder='/research/astro/fir/HELP/DESPHOT/'
folder='/Users/pdh21/HELP/XID_plus_output/plot_test/'
hdulist=fits.open(folder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise__PSWXID_S100_50mic_test.fits')
fcat=hdulist[1].data
nsources_xid=fcat.shape[0]
print nsources_xid
hdulist.close()

#---Read in truth catalogue---
hdulist=fits.open(folder+'lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS_test_pacs100_50cut.fits')
fcat_sim=hdulist[1].data
hdulist.close()

#-----set up truncated boundaries for DESPHOT----
low_clip=0.0
up_clip=1000.0
flattened_post_psw=np.empty((1000,nsources_xid))
flattened_post_pmw=np.empty((1000,nsources_xid))
flattened_post_plw=np.empty((1000,nsources_xid))

from scipy.stats import truncnorm
for i in range(0,nsources_xid):
    my_mean=fcat['F250'][i]
    my_std=fcat['E250'][i]
    a,b = (low_clip - my_mean)/my_std,(up_clip - my_mean)/my_std
    flattened_post_psw[:,i] = (truncnorm.rvs(a, b, size=1000)*my_std)+my_mean
    my_mean=fcat['F350'][i]
    my_std=fcat['E350'][i]
    a,b = (low_clip - my_mean)/my_std,(up_clip - my_mean)/my_std
    flattened_post_pmw[:,i] = (truncnorm.rvs(a, b, size=1000)*my_std)+my_mean
    my_mean=fcat['F500'][i]
    my_std=fcat['E500'][i]
    a,b = (low_clip - my_mean)/my_std,(up_clip - my_mean)/my_std
    flattened_post_plw[:,i] = (truncnorm.rvs(a, b, size=1000)*my_std)+my_mean    

ind_1mjy_psw=fcat['F250']> 1
ind_1mjy_pmw=fcat['F350']> 1
ind_1mjy_plw=fcat['F500']> 1

psw_metrics_XIDp=metrics_XIDp(flattened_post_psw[:,ind_1mjy_psw],fcat_sim['S250'][ind_1mjy_psw])
pmw_metrics_XIDp=metrics_XIDp(flattened_post_pmw[:,ind_1mjy_pmw],fcat_sim['S350'][ind_1mjy_pmw])
plw_metrics_XIDp=metrics_XIDp(flattened_post_plw[:,ind_1mjy_plw],fcat_sim['S500'][ind_1mjy_plw])

#ind_z_strange=psw_metrics_XIDp[0]>2.9
#print psw_metrics_XIDp[0][ind_z_strange]
#print fcat['F250'][ind_1mjy_psw][ind_z_strange]
#print fcat_sim['S250'][ind_1mjy_psw][ind_z_strange]
#plt.plot(psw_metrics_XIDp[0],fcat_sim['S250'][ind_1mjy_psw],'o')
#quit()




bins=np.logspace(0.477, 2.2, num=7)
labels=[r'Z score', r'IQR/$S_{True}$', r'$(S_{Obs}-S_{True})/S_{True}$']
scale=['linear', 'log', 'linear']
ylims=[(-4,4),(1E-2,1E1),(-1,1)]
for i in range(0,3):
    pdf_pages.savefig(metrics_plot(psw_metrics_XIDp[i],fcat_sim['S250'][ind_1mjy_psw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i]))
    pdf_pages.savefig(metrics_plot(pmw_metrics_XIDp[i],fcat_sim['S350'][ind_1mjy_pmw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i],cmap=plt.get_cmap('Greens')))
    pdf_pages.savefig(metrics_plot(plw_metrics_XIDp[i],fcat_sim['S500'][ind_1mjy_plw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i],cmap=plt.get_cmap('Reds')))

pdf_pages.close()
