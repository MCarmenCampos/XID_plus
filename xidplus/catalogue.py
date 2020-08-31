import numpy as np
from astropy.io import fits
import xidplus.io as io
import xidplus.posterior_maps as postmaps




def create_PACS_cat(posterior, prior100, prior160):

    """
    Create PACS catalogue from posterior
    
    :param posterior: PACS xidplus.posterior class
    :param prior100:  PACS 100 xidplus.prior class
    :param prior160:  PACS 160 xidplus.prior class
    :return: fits hdulist
    """
    import datetime
    nsrc=prior100.nsrc
    rep_maps=postmaps.replicated_maps([prior100,prior160],posterior)
    Bayes_P100=postmaps.Bayes_Pval_res(prior100,rep_maps[0])
    Bayes_P160=postmaps.Bayes_Pval_res(prior160,rep_maps[1])


    # ----table info-----------------------
    # first define columns
    c1 = fits.Column(name='help_id', format='27A', array=prior100.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior100.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior100.sdec)
    c4 = fits.Column(name='F_PACS_100', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_PACS_100_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_PACS_100_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_PACS_160', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_PACS_160_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_PACS_160_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='Bkg_PACS_100', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c11 = fits.Column(name='Bkg_PACS_160', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,1],50.0,axis=0)))
    c12 = fits.Column(name='Sig_conf_PACS_100', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c13 = fits.Column(name='Sig_conf_PACS_160', format='E', unit='mJy/Beam',
                      array=np.full(nsrc, np.percentile(posterior.samples['sigma_conf'][:,1],50.0,axis=0)))
    c14 = fits.Column(name='Rhat_PACS_100', format='E', array=posterior.Rhat['src_f'][:,0])
    c15 = fits.Column(name='Rhat_PACS_160', format='E', array=posterior.Rhat['src_f'][:,1])
    c16 = fits.Column(name='n_eff_PACS_100', format='E', array=posterior.n_eff['src_f'][:,0])
    c17 = fits.Column(name='n_eff_PACS_160', format='E', array=posterior.n_eff['src_f'][:,1])
    c18 = fits.Column(name='Pval_res_100', format='E', array=Bayes_P100)
    c19 = fits.Column(name='Pval_res_160', format='E', array=Bayes_P160)


    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19])

    tbhdu.header.set('TUCD1', 'ID', after='TFORM1')
    tbhdu.header.set('TDESC1', 'ID of source', after='TUCD1')

    tbhdu.header.set('TUCD2', 'pos.eq.RA', after='TUNIT2')
    tbhdu.header.set('TDESC2', 'R.A. of object J2000', after='TUCD2')

    tbhdu.header.set('TUCD3', 'pos.eq.DEC', after='TUNIT3')
    tbhdu.header.set('TDESC3', 'Dec. of object J2000', after='TUCD3')

    tbhdu.header.set('TUCD4', 'phot.flux.density', after='TUNIT4')
    tbhdu.header.set('TDESC4', '100 Flux (at 50th percentile)', after='TUCD4')

    tbhdu.header.set('TUCD5', 'phot.flux.density', after='TUNIT5')
    tbhdu.header.set('TDESC5', '100 Flux (at 84.1 percentile) ', after='TUCD5')

    tbhdu.header.set('TUCD6', 'phot.flux.density', after='TUNIT6')
    tbhdu.header.set('TDESC6', '100 Flux (at 15.9 percentile)', after='TUCD6')

    tbhdu.header.set('TUCD7', 'phot.flux.density', after='TUNIT7')
    tbhdu.header.set('TDESC7', '160 Flux (at 50th percentile)', after='TUCD7')

    tbhdu.header.set('TUCD8', 'phot.flux.density', after='TUNIT8')
    tbhdu.header.set('TDESC8', '160 Flux (at 84.1 percentile) ', after='TUCD8')

    tbhdu.header.set('TUCD9', 'phot.flux.density', after='TUNIT9')
    tbhdu.header.set('TDESC9', '160 Flux (at 15.9 percentile)', after='TUCD9')

    tbhdu.header.set('TUCD10', 'phot.flux.density', after='TUNIT10')
    tbhdu.header.set('TDESC10', '100 background', after='TUCD10')

    tbhdu.header.set('TUCD11', 'phot.flux.density', after='TUNIT11')
    tbhdu.header.set('TDESC11', '160 background', after='TUCD11')

    tbhdu.header.set('TUCD12', 'phot.flux.density', after='TUNIT12')
    tbhdu.header.set('TDESC12', '100 residual confusion noise', after='TUCD12')

    tbhdu.header.set('TUCD13', 'phot.flux.density', after='TUNIT13')
    tbhdu.header.set('TDESC13', '160 residual confusion noise', after='TUCD13')

    tbhdu.header.set('TUCD14', 'stat.value', after='TFORM14')
    tbhdu.header.set('TDESC14', '100 MCMC Convergence statistic', after='TUCD14')

    tbhdu.header.set('TUCD15', 'stat.value', after='TFORM15')
    tbhdu.header.set('TDESC15', '160 MCMC Convergence statistic', after='TUCD15')

    tbhdu.header.set('TUCD16', 'stat.value', after='TFORM16')
    tbhdu.header.set('TDESC16', '100 MCMC independence statistic', after='TUCD16')

    tbhdu.header.set('TUCD17', 'stat.value', after='TFORM17')
    tbhdu.header.set('TDESC17', '160 MCMC independence statistic', after='TUCD17')
    
    tbhdu.header.set('TUCD18','stat.value',after='TFORM18')
    tbhdu.header.set('TDESC18','100 Bayes Pval residual statistic',after='TUCD18')

    tbhdu.header.set('TUCD19','stat.value',after='TFORM19')
    tbhdu.header.set('TDESC19','160 Bayes Pval residual statistic',after='TUCD19')
    # ----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior100.prior_cat
    prihdr['TITLE'] = 'PACS XID+ catalogue'
    # prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change
    prihdr['CREATOR'] = 'WP5'
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE'] = datetime.datetime.now().isoformat()
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist

# noinspection PyPackageRequirements

def create_MIPS_cat(posterior, prior24, Bayes_P24):

    """
    Create MIPS catalogue from posterior
    
    :param posterior: MIPS xidplus.posterior class
    :param prior24: MIPS xidplus.prior class
    :param Bayes_P24:  Bayes Pvalue residual statistic for MIPS 24
    :return: fits hdulist
    """
    import datetime
    nsrc=prior24.nsrc
    rep_maps = postmaps.replicated_maps([prior24], posterior)
    Bayes_P24 = postmaps.Bayes_Pval_res(prior24, rep_maps[0])
    # ----table info-----------------------
    # first define columns
    c1 = fits.Column(name='help_id', format='27A', array=prior24.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior24.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior24.sdec)
    c4 = fits.Column(name='F_MIPS_24', format='E', unit='muJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_MIPS_24_u', format='E', unit='muJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_MIPS_24_l', format='E', unit='muJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='Bkg_MIPS_24', format='E', unit='MJy/sr',
                     array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c8 = fits.Column(name='Sig_conf_MIPS_24', format='E', unit='MJy/sr',
                     array=np.full(nsrc, np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c9 = fits.Column(name='Rhat_MIPS_24', format='E', array=posterior.Rhat['src_f'][:,0])
    c10 = fits.Column(name='n_eff_MIPS_24', format='E', array=posterior.n_eff['src_f'][:,0])
    c11 = fits.Column(name='Pval_res_24', format='E', array=Bayes_P24)
    
    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11])

    tbhdu.header.set('TUCD1', 'ID', after='TFORM1')
    tbhdu.header.set('TDESC1', 'ID of source', after='TUCD1')

    tbhdu.header.set('TUCD2', 'pos.eq.RA', after='TUNIT2')
    tbhdu.header.set('TDESC2', 'R.A. of object J2000', after='TUCD2')

    tbhdu.header.set('TUCD3', 'pos.eq.DEC', after='TUNIT3')
    tbhdu.header.set('TDESC3', 'Dec. of object J2000', after='TUCD3')

    tbhdu.header.set('TUCD4', 'phot.flux.density', after='TUNIT4')
    tbhdu.header.set('TDESC4', '24 Flux (at 50th percentile)', after='TUCD4')

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')
    tbhdu.header.set('TDESC5','24 Flux (at 84.1 percentile) ',after='TUCD5')

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')
    tbhdu.header.set('TDESC6','24 Flux (at 15.9 percentile)',after='TUCD6')

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')
    tbhdu.header.set('TDESC7','24 background',after='TUCD7')

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')
    tbhdu.header.set('TDESC8','24 residual confusion noise',after='TUCD8')

    tbhdu.header.set('TUCD9','stat.value',after='TFORM9')
    tbhdu.header.set('TDESC9','24 MCMC Convergence statistic',after='TUCD9')

    tbhdu.header.set('TUCD10','stat.value',after='TFORM10')
    tbhdu.header.set('TDESC10','24 MCMC independence statistic',after='TUCD10')

    tbhdu.header.set('TUCD11','stat.value',after='TFORM11')
    tbhdu.header.set('TDESC11','24 Bayes Pval residual statistic',after='TUCD11')

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior24.prior_cat
    prihdr['TITLE']   = 'XID+MIPS catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change
    prihdr['CREATOR'] = 'WP5'
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist
# noinspection PyPackageRequirements


def create_SPIRE_cat(posterior,prior250,prior350,prior500):

    """
    Create SPIRE catalogue from posterior


    :param posterior: SPIRE xidplus.posterior class
    :param prior250: SPIRE 250 xidplus.prior class
    :param prior350: SPIRE 350 xidplus.prior class
    :param prior500: SPIRE 500 xidplus.prior class
    :param Bayes_P250: Bayes Pvalue residual statistic for SPIRE 250
    :param Bayes_P350: Bayes Pvalue residual statistic for SPIRE 350
    :param Bayes_P500: Bayes Pvalue residual statistic for SPIRE 500
    :return: fits hdulist
    """
    import datetime
    nsrc=posterior.nsrc
    rep_maps = postmaps.replicated_maps([prior250, prior350,prior500], posterior)
    Bayes_P250 = postmaps.Bayes_Pval_res(prior250, rep_maps[0])
    Bayes_P350 = postmaps.Bayes_Pval_res(prior350, rep_maps[1])
    Bayes_P500 = postmaps.Bayes_Pval_res(prior500, rep_maps[2])


    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='HELP_ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='Bkg_SPIRE_250', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c14 = fits.Column(name='Bkg_SPIRE_350', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,1],50.0,axis=0)))
    c15 = fits.Column(name='Bkg_SPIRE_500', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,2],50.0,axis=0)))
    c16 = fits.Column(name='Sig_conf_SPIRE_250', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c17 = fits.Column(name='Sig_conf_SPIRE_350', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,1],50.0,axis=0)))
    c18 = fits.Column(name='Sig_conf_SPIRE_500', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,2],50.0,axis=0)))
    c19 = fits.Column(name='Rhat_SPIRE_250', format='E', array=posterior.Rhat['src_f'][:,0])
    c20 = fits.Column(name='Rhat_SPIRE_350', format='E', array=posterior.Rhat['src_f'][:,1])
    c21 = fits.Column(name='Rhat_SPIRE_500', format='E', array=posterior.Rhat['src_f'][:,2])
    c22 = fits.Column(name='n_eff_SPIRE_250', format='E', array=posterior.n_eff['src_f'][:,0])
    c23 = fits.Column(name='n_eff_SPIRE_350', format='E', array=posterior.n_eff['src_f'][:,1])
    c24 = fits.Column(name='n_eff_SPIRE_500', format='E', array=posterior.n_eff['src_f'][:,2])
    c25 = fits.Column(name='Pval_res_250', format='E', array=Bayes_P250)
    c26 = fits.Column(name='Pval_res_350', format='E', array=Bayes_P350)
    c27 = fits.Column(name='Pval_res_500', format='E', array=Bayes_P500)
    

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22,
                            c24, c23, c25, c26, c27])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','250 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','250 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','250 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','350 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','350 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','350 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','500 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','500 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','500 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 background',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','350 background',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','500 background',after='TUCD15')

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')
    tbhdu.header.set('TDESC16','250 residual confusion noise',after='TUCD16')

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')
    tbhdu.header.set('TDESC17','350 residual confusion noise',after='TUCD17')

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')
    tbhdu.header.set('TDESC18','500 residual confusion noise',after='TUCD18')

    tbhdu.header.set('TUCD19','stat.value',after='TFORM16')
    tbhdu.header.set('TDESC19','250 MCMC Convergence statistic',after='TUCD19')

    tbhdu.header.set('TUCD20','stat.value',after='TFORM20')
    tbhdu.header.set('TDESC20','350 MCMC Convergence statistic',after='TUCD20')

    tbhdu.header.set('TUCD21','stat.value',after='TFORM21')
    tbhdu.header.set('TDESC21','500 MCMC Convergence statistic',after='TUCD21')

    tbhdu.header.set('TUCD22','stat.value',after='TFORM22')
    tbhdu.header.set('TDESC22','250 MCMC independence statistic',after='TUCD22')

    tbhdu.header.set('TUCD23','stat.value',after='TFORM23')
    tbhdu.header.set('TDESC23','350 MCMC independence statistic',after='TUCD23')

    tbhdu.header.set('TUCD24','stat.value',after='TFORM24')
    tbhdu.header.set('TDESC24','500 MCMC independence statistic',after='TUCD24')

    tbhdu.header.set('TUCD25','stat.value',after='TFORM25')
    tbhdu.header.set('TDESC25','250 Bayes Pval residual statistic',after='TUCD25')

    tbhdu.header.set('TUCD26','stat.value',after='TFORM26')
    tbhdu.header.set('TDESC26','350 Bayes Pval residual statistic',after='TUCD26')

    tbhdu.header.set('TUCD27','stat.value',after='TFORM27')
    tbhdu.header.set('TDESC27','500 Bayes Pval residual statistic',after='TUCD27')

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'SPIRE XID+ catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist


def create_LABOCA_cat(posterior, prior870):

    """
    Create LABOCA catalogue from posterior
    
    :param posterior: LABOCA xidplus.posterior class
    :param prior870: LABOCA xidplus.prior class
    :param Bayes_P870:  Bayes Pvalue residual statistic for LABOCA 870
    :return: fits hdulist
    """
    import datetime
    nsrc=prior870.nsrc
    rep_maps = postmaps.replicated_maps([prior870], posterior)
    Bayes_P870 = postmaps.Bayes_Pval_res(prior870, rep_maps[0])
    # ----table info-----------------------
    # first define columns
    c1 = fits.Column(name='help_id', format='27A', array=prior870.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior870.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior870.sdec)
    c4 = fits.Column(name='F_LABOCA_850', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_LABOCA_850_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_LABOCA_850_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='Bkg_LABOCA_850', format='E', unit='Jy/beam',
                     array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c8 = fits.Column(name='Sig_conf_LABOCA_850', format='E', unit='Jy/beam',
                     array=np.full(nsrc, np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c9 = fits.Column(name='Rhat_LABOCA_850', format='E', array=posterior.Rhat['src_f'][:,0])
    c10 = fits.Column(name='n_eff_LABOCA_850', format='E', array=posterior.n_eff['src_f'][:,0])
    c11 = fits.Column(name='Pval_res_870', format='E', array=Bayes_P870)
    
    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11])

    tbhdu.header.set('TUCD1', 'ID', after='TFORM1')
    tbhdu.header.set('TDESC1', 'ID of source', after='TUCD1')

    tbhdu.header.set('TUCD2', 'pos.eq.RA', after='TUNIT2')
    tbhdu.header.set('TDESC2', 'R.A. of object J2000', after='TUCD2')

    tbhdu.header.set('TUCD3', 'pos.eq.DEC', after='TUNIT3')
    tbhdu.header.set('TDESC3', 'Dec. of object J2000', after='TUCD3')

    tbhdu.header.set('TUCD4', 'phot.flux.density', after='TUNIT4')
    tbhdu.header.set('TDESC4', '870 Flux (at 50th percentile)', after='TUCD4')

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')
    tbhdu.header.set('TDESC5','870 Flux (at 84.1 percentile) ',after='TUCD5')

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')
    tbhdu.header.set('TDESC6','870 Flux (at 15.9 percentile)',after='TUCD6')

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')
    tbhdu.header.set('TDESC7','870 background',after='TUCD7')

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')
    tbhdu.header.set('TDESC8','870 residual confusion noise',after='TUCD8')

    tbhdu.header.set('TUCD9','stat.value',after='TFORM9')
    tbhdu.header.set('TDESC9','870 MCMC Convergence statistic',after='TUCD9')

    tbhdu.header.set('TUCD10','stat.value',after='TFORM10')
    tbhdu.header.set('TDESC10','870 MCMC independence statistic',after='TUCD10')

    tbhdu.header.set('TUCD11','stat.value',after='TFORM11')
    tbhdu.header.set('TDESC11','870 Bayes Pval residual statistic',after='TUCD11')

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior870.prior_cat
    prihdr['TITLE']   = 'XID+LABOCA catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change
    prihdr['CREATOR'] = 'WP5'
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist
# noinspection PyPackageRequirements


def create_SPIRE_SED_cat(posterior,prior250,prior350,prior500):

    """
    Create SPIRE catalogue from posterior


    :param posterior: SPIRE xidplus.posterior class
    :param prior250: SPIRE 250 xidplus.prior class
    :param prior350: SPIRE 350 xidplus.prior class
    :param prior500: SPIRE 500 xidplus.prior class
    :param Bayes_P250: Bayes Pvalue residual statistic for SPIRE 250
    :param Bayes_P350: Bayes Pvalue residual statistic for SPIRE 350
    :param Bayes_P500: Bayes Pvalue residual statistic for SPIRE 500
    :return: fits hdulist
    """
    import datetime
    nsrc=posterior.nsrc
    rep_maps = postmaps.replicated_maps([prior250, prior350,prior500], posterior)
    Bayes_P250 = postmaps.Bayes_Pval_res(prior250, rep_maps[0])
    Bayes_P350 = postmaps.Bayes_Pval_res(prior350, rep_maps[1])
    Bayes_P500 = postmaps.Bayes_Pval_res(prior500, rep_maps[2])


    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='HELP_ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='Bkg_SPIRE_250', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c14 = fits.Column(name='Bkg_SPIRE_350', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,1],50.0,axis=0)))
    c15 = fits.Column(name='Bkg_SPIRE_500', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,2],50.0,axis=0)))
    c16 = fits.Column(name='Sig_conf_SPIRE_250', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c17 = fits.Column(name='Sig_conf_SPIRE_350', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,1],50.0,axis=0)))
    c18 = fits.Column(name='Sig_conf_SPIRE_500', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,2],50.0,axis=0)))
    c19 = fits.Column(name='Rhat_SPIRE_250', format='E', array=posterior.Rhat['src_f'][:,0])
    c20 = fits.Column(name='Rhat_SPIRE_350', format='E', array=posterior.Rhat['src_f'][:,1])
    c21 = fits.Column(name='Rhat_SPIRE_500', format='E', array=posterior.Rhat['src_f'][:,2])
    c22 = fits.Column(name='n_eff_SPIRE_250', format='E', array=posterior.n_eff['src_f'][:,0])
    c23 = fits.Column(name='n_eff_SPIRE_350', format='E', array=posterior.n_eff['src_f'][:,1])
    c24 = fits.Column(name='n_eff_SPIRE_500', format='E', array=posterior.n_eff['src_f'][:,2])
    c25 = fits.Column(name='Pval_res_250', format='E', array=Bayes_P250)
    c26 = fits.Column(name='Pval_res_350', format='E', array=Bayes_P350)
    c27 = fits.Column(name='Pval_res_500', format='E', array=Bayes_P500)
    
    
    c28 = fits.Column(name='z', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],50.0,axis=0))
    c29 = fits.Column(name='zErr_u', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],84.1,axis=0))
    c30 = fits.Column(name='zErr_l', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],15.9,axis=0))
    c31 = fits.Column(name='Nbb', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],50.0,axis=0))
    c32 = fits.Column(name='Nbb_u', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],84.1,axis=0))
    c33 = fits.Column(name='Nbb_l', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],15.9,axis=0))
    
    

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22,
                            c24, c23, c25, c26, c27, c28, c29, c30, c31, c32, c33])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','250 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','250 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','250 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','350 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','350 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','350 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','500 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','500 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','500 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 background',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','350 background',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','500 background',after='TUCD15')

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')
    tbhdu.header.set('TDESC16','250 residual confusion noise',after='TUCD16')

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')
    tbhdu.header.set('TDESC17','350 residual confusion noise',after='TUCD17')

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')
    tbhdu.header.set('TDESC18','500 residual confusion noise',after='TUCD18')

    tbhdu.header.set('TUCD19','stat.value',after='TFORM16')
    tbhdu.header.set('TDESC19','250 MCMC Convergence statistic',after='TUCD19')

    tbhdu.header.set('TUCD20','stat.value',after='TFORM20')
    tbhdu.header.set('TDESC20','350 MCMC Convergence statistic',after='TUCD20')

    tbhdu.header.set('TUCD21','stat.value',after='TFORM21')
    tbhdu.header.set('TDESC21','500 MCMC Convergence statistic',after='TUCD21')

    tbhdu.header.set('TUCD22','stat.value',after='TFORM22')
    tbhdu.header.set('TDESC22','250 MCMC independence statistic',after='TUCD22')

    tbhdu.header.set('TUCD23','stat.value',after='TFORM23')
    tbhdu.header.set('TDESC23','350 MCMC independence statistic',after='TUCD23')

    tbhdu.header.set('TUCD24','stat.value',after='TFORM24')
    tbhdu.header.set('TDESC24','500 MCMC independence statistic',after='TUCD24')

    tbhdu.header.set('TUCD25','stat.value',after='TFORM25')
    tbhdu.header.set('TDESC25','250 Bayes Pval residual statistic',after='TUCD25')

    tbhdu.header.set('TUCD26','stat.value',after='TFORM26')
    tbhdu.header.set('TDESC26','350 Bayes Pval residual statistic',after='TUCD26')

    tbhdu.header.set('TUCD27','stat.value',after='TFORM27')
    tbhdu.header.set('TDESC27','500 Bayes Pval residual statistic',after='TUCD27')
    
    
    tbhdu.header.set('TUCD28','phot.redshift',after='TFORM28')      
    tbhdu.header.set('TDESC28','phot.redshift (at 50th percentile)',after='TUCD28') 

    tbhdu.header.set('TUCD29','phot.redshift',after='TFORM29')      
    tbhdu.header.set('TDESC29','phot.redshift (at 84.1 percentile) ',after='TUCD29') 
    
    tbhdu.header.set('TUCD30','phot.redshift',after='TFORM30')      
    tbhdu.header.set('TDESC30','phot.redshift (at 15.9 percentile)',after='TUCD30')

    tbhdu.header.set('TUCD31','log.Lsol',after='TUNIT31')      
    tbhdu.header.set('TDESC31','log(Luminosity) (at 50th percentile)',after='TUCD31') 

    tbhdu.header.set('TUCD32','log.Lsol',after='TUNIT32')      
    tbhdu.header.set('TDESC32','log(Luminosity) (at 84.1 percentile) ',after='TUCD32') 

    tbhdu.header.set('TUCD33','log.Lsol',after='TUNIT33')      
    tbhdu.header.set('TDESC33','log(Luminosity) (at 15.9 percentile)',after='TUCD33')

    

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'SPIRE XID+ catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist



def create_IR_cat(posterior,priors):

    """
    Create SED catalogue from posterior


    :param posterior: SED xidplus.posterior class
    :param priors: MIPS-PACS-SPIRE-LABOCA xidplus.prior class
    :return: fits hdulist
    """
    import datetime
    nsrc=posterior.nsrc
    prior24  = priors[0]
    prior100 = priors[1]
    prior160 = priors[2]
    prior250 = priors[3]
    prior350 = priors[4]
    prior500 = priors[5]

    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_MIPS_24', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_MIPS_24_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_MIPS_24_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_PACS_100', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_PACS_100_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_PACS_100_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_PACS_160', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_PACS_160_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_PACS_160_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],50.0,axis=0))
    c14 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],84.1,axis=0))
    c15 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],15.9,axis=0))
    c16 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],50.0,axis=0))
    c17 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],84.1,axis=0))
    c18 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],15.9,axis=0))
    c19 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],50.0,axis=0))
    c20 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],84.1,axis=0))
    c21 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],15.9,axis=0))
    c22 = fits.Column(name='z', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],50.0,axis=0))
    c23 = fits.Column(name='zErr_u', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],84.1,axis=0))
    c24 = fits.Column(name='zErr_l', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],15.9,axis=0))
    c25 = fits.Column(name='Nbb', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],50.0,axis=0))
    c26 = fits.Column(name='Nbb_u', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],84.1,axis=0))
    c27 = fits.Column(name='Nbb_l', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],15.9,axis=0))
    

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12, c13, c14, c15, c16, c17, c18, c19, c20, c21,
                            c22, c23, c24, c25, c26, c27])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','24 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','24 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','24 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','100 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','100 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','100 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','160 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','160 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','160 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 Flux (at 50th percentile)',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','250 Flux (at 84.1 percentile) ',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','250 Flux (at 15.9 percentile)',after='TUCD15') 

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')      
    tbhdu.header.set('TDESC16','350 Flux (at 50th percentile)',after='TUCD16') 

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')      
    tbhdu.header.set('TDESC17','350 Flux (at 84.1 percentile) ',after='TUCD17') 

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')      
    tbhdu.header.set('TDESC18','350 Flux (at 15.9 percentile)',after='TUCD18') 

    tbhdu.header.set('TUCD19','phot.flux.density',after='TUNIT19')      
    tbhdu.header.set('TDESC19','500 Flux (at 50th percentile)',after='TUCD19') 

    tbhdu.header.set('TUCD20','phot.flux.density',after='TUNIT20')      
    tbhdu.header.set('TDESC20','500 Flux (at 84.1 percentile) ',after='TUCD20') 

    tbhdu.header.set('TUCD21','phot.flux.density',after='TUNIT21')      
    tbhdu.header.set('TDESC21','500 Flux (at 15.9 percentile)',after='TUCD21')
    
    tbhdu.header.set('TUCD22','phot.redshift',after='TFORM22')      
    tbhdu.header.set('TDESC22','phot.redshift (at 50th percentile)',after='TUCD22') 

    tbhdu.header.set('TUCD23','phot.redshift',after='TFORM23')      
    tbhdu.header.set('TDESC23','phot.redshift (at 84.1 percentile) ',after='TUCD23') 
    
    tbhdu.header.set('TUCD24','phot.redshift',after='TFORM24')      
    tbhdu.header.set('TDESC24','phot.redshift (at 15.9 percentile)',after='TUCD24')

    tbhdu.header.set('TUCD25','log.Lsol',after='TUNIT25')      
    tbhdu.header.set('TDESC25','log(Luminosity) (at 50th percentile)',after='TUCD25') 

    tbhdu.header.set('TUCD26','log.Lsol',after='TUNIT26')      
    tbhdu.header.set('TDESC26','log(Luminosity) (at 84.1 percentile) ',after='TUCD26') 

    tbhdu.header.set('TUCD27','log.Lsol',after='TUNIT27')      
    tbhdu.header.set('TDESC27','log(Luminosity) (at 15.9 percentile)',after='TUCD27')

    
    

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'XID+SED catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist


def create_IR_cat_fixz(posterior,priors):

    """
    Create SED catalogue from posterior


    :param posterior: SED xidplus.posterior class
    :param priors: MIPS-PACS-SPIRE-LABOCA xidplus.prior class
    :return: fits hdulist
    """
    import datetime
    nsrc=posterior.nsrc
    prior24  = priors[0]
    prior100 = priors[1]
    prior160 = priors[2]
    prior250 = priors[3]
    prior350 = priors[4]
    prior500 = priors[5]

    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_MIPS_24', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_MIPS_24_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_MIPS_24_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_PACS_100', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_PACS_100_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_PACS_100_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_PACS_160', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_PACS_160_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_PACS_160_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],50.0,axis=0))
    c14 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],84.1,axis=0))
    c15 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],15.9,axis=0))
    c16 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],50.0,axis=0))
    c17 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],84.1,axis=0))
    c18 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],15.9,axis=0))
    c19 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],50.0,axis=0))
    c20 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],84.1,axis=0))
    c21 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],15.9,axis=0))
    c22 = fits.Column(name='Nbb', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],50.0,axis=0))
    c23 = fits.Column(name='Nbb_u', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],84.1,axis=0))
    c24 = fits.Column(name='Nbb_l', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],15.9,axis=0))
    

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12, c13, c14, c15, c16, c17, c18, c19, c20, c21,
                            c22, c23, c24])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','24 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','24 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','24 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','100 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','100 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','100 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','160 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','160 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','160 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 Flux (at 50th percentile)',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','250 Flux (at 84.1 percentile) ',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','250 Flux (at 15.9 percentile)',after='TUCD15') 

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')      
    tbhdu.header.set('TDESC16','350 Flux (at 50th percentile)',after='TUCD16') 

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')      
    tbhdu.header.set('TDESC17','350 Flux (at 84.1 percentile) ',after='TUCD17') 

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')      
    tbhdu.header.set('TDESC18','350 Flux (at 15.9 percentile)',after='TUCD18') 

    tbhdu.header.set('TUCD19','phot.flux.density',after='TUNIT19')      
    tbhdu.header.set('TDESC19','500 Flux (at 50th percentile)',after='TUCD19') 

    tbhdu.header.set('TUCD20','phot.flux.density',after='TUNIT20')      
    tbhdu.header.set('TDESC20','500 Flux (at 84.1 percentile) ',after='TUCD20') 

    tbhdu.header.set('TUCD21','phot.flux.density',after='TUNIT21')      
    tbhdu.header.set('TDESC21','500 Flux (at 15.9 percentile)',after='TUCD21')

    tbhdu.header.set('TUCD22','log.Lsol',after='TUNIT22')      
    tbhdu.header.set('TDESC22','log(Luminosity) (at 50th percentile)',after='TUCD22') 

    tbhdu.header.set('TUCD23','log.Lsol',after='TUNIT23')      
    tbhdu.header.set('TDESC23','log(Luminosity) (at 84.1 percentile) ',after='TUCD23') 

    tbhdu.header.set('TUCD24','log.Lsol',after='TUNIT24')      
    tbhdu.header.set('TDESC24','log(Luminosity) (at 15.9 percentile)',after='TUCD24')

    
    

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'XID+SED catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist



def create_SPIRE_gaussprior_cat(posterior,priors):

    """
    Create SED catalogue from posterior


    :param posterior: SED xidplus.posterior class
    :param priors: MIPS-PACS-SPIRE-LABOCA xidplus.prior class
    :return: fits hdulist
    """
    
    import datetime
    nsrc=posterior.nsrc
    prior24  = priors[0]
    prior100 = priors[1]
    prior160 = priors[2]
    prior250 = priors[3]
    prior350 = priors[4]
    prior500 = priors[5]
    prior850 = priors[6]
    
    rep_maps = postmaps.replicated_maps([prior250, prior350,prior500], posterior)
    Bayes_P250 = postmaps.Bayes_Pval_res(prior250, rep_maps[0])
    Bayes_P350 = postmaps.Bayes_Pval_res(prior350, rep_maps[1])
    Bayes_P500 = postmaps.Bayes_Pval_res(prior500, rep_maps[2])

    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_MIPS_24', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_MIPS_24_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_MIPS_24_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_PACS_100', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_PACS_100_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_PACS_100_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_PACS_160', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_PACS_160_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_PACS_160_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],50.0,axis=0))
    c14 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],84.1,axis=0))
    c15 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],15.9,axis=0))
    c16 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],50.0,axis=0))
    c17 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],84.1,axis=0))
    c18 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],15.9,axis=0))
    c19 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],50.0,axis=0))
    c20 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],84.1,axis=0))
    c21 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],15.9,axis=0))
    c22 = fits.Column(name='F_LABOCA_850', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],50.0,axis=0))
    c23 = fits.Column(name='FErr_LABOCA_850_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],84.1,axis=0))
    c24 = fits.Column(name='FErr_LABOCA_850_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],15.9,axis=0))
    
    
    c25 = fits.Column(name='Bkg_SPIRE_250', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c26 = fits.Column(name='Bkg_SPIRE_350', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,1],50.0,axis=0)))
    c27 = fits.Column(name='Bkg_SPIRE_500', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,2],50.0,axis=0)))
    
    c28 = fits.Column(name='Sig_conf_SPIRE_250', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c29 = fits.Column(name='Sig_conf_SPIRE_350', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,1],50.0,axis=0)))
    c30 = fits.Column(name='Sig_conf_SPIRE_500', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,2],50.0,axis=0)))
    
    c31 = fits.Column(name='z', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],50.0,axis=0))
    c32 = fits.Column(name='zErr_u', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],84.1,axis=0))
    c33 = fits.Column(name='zErr_l', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],15.9,axis=0))
    c34 = fits.Column(name='Nbb', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],50.0,axis=0))
    c35 = fits.Column(name='Nbb_u', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],84.1,axis=0))
    c36 = fits.Column(name='Nbb_l', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],15.9,axis=0))
    
    
    c37 = fits.Column(name='Rhat_SPIRE_250', format='E', array=posterior.Rhat['src_f'][:,0])
    c38 = fits.Column(name='Rhat_SPIRE_350', format='E', array=posterior.Rhat['src_f'][:,1])
    c39 = fits.Column(name='Rhat_SPIRE_500', format='E', array=posterior.Rhat['src_f'][:,2])
    c40 = fits.Column(name='n_eff_SPIRE_250', format='E', array=posterior.n_eff['src_f'][:,0])
    c41 = fits.Column(name='n_eff_SPIRE_350', format='E', array=posterior.n_eff['src_f'][:,1])
    c42 = fits.Column(name='n_eff_SPIRE_500', format='E', array=posterior.n_eff['src_f'][:,2])
    c43 = fits.Column(name='Pval_res_250', format='E', array=Bayes_P250)
    c44 = fits.Column(name='Pval_res_350', format='E', array=Bayes_P350)
    c45 = fits.Column(name='Pval_res_500', format='E', array=Bayes_P500)

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,
                                           c28,c29,c30,
                                           c31,c32,c33,c34,c35,c36,
                                           c37,c38,
                                           c39,c40,c41,c42,c43,c44,c45])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','24 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','24 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','24 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','100 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','100 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','100 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','160 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','160 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','160 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 Flux (at 50th percentile)',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','250 Flux (at 84.1 percentile) ',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','250 Flux (at 15.9 percentile)',after='TUCD15') 

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')      
    tbhdu.header.set('TDESC16','350 Flux (at 50th percentile)',after='TUCD16') 

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')      
    tbhdu.header.set('TDESC17','350 Flux (at 84.1 percentile) ',after='TUCD17') 

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')      
    tbhdu.header.set('TDESC18','350 Flux (at 15.9 percentile)',after='TUCD18') 

    tbhdu.header.set('TUCD19','phot.flux.density',after='TUNIT19')      
    tbhdu.header.set('TDESC19','500 Flux (at 50th percentile)',after='TUCD19') 

    tbhdu.header.set('TUCD20','phot.flux.density',after='TUNIT20')      
    tbhdu.header.set('TDESC20','500 Flux (at 84.1 percentile) ',after='TUCD20') 

    tbhdu.header.set('TUCD21','phot.flux.density',after='TUNIT21')      
    tbhdu.header.set('TDESC21','500 Flux (at 15.9 percentile)',after='TUCD21')

    tbhdu.header.set('TUCD22','phot.flux.density',after='TUNIT22')      
    tbhdu.header.set('TDESC22','850 Flux (at 50th percentile)',after='TUCD22') 

    tbhdu.header.set('TUCD23','phot.flux.density',after='TUNIT23')      
    tbhdu.header.set('TDESC23','850 Flux (at 84.1 percentile) ',after='TUCD23') 

    tbhdu.header.set('TUCD24','phot.flux.density',after='TUNIT24')      
    tbhdu.header.set('TDESC24','850 Flux (at 15.9 percentile)',after='TUCD24')
    
    tbhdu.header.set('TUCD25','phot.flux.density',after='TUNIT25')      
    tbhdu.header.set('TDESC25','250 background',after='TUCD25') 

    tbhdu.header.set('TUCD26','phot.flux.density',after='TUNIT26')      
    tbhdu.header.set('TDESC26','350 background',after='TUCD26') 

    tbhdu.header.set('TUCD27','phot.flux.density',after='TUNIT27')      
    tbhdu.header.set('TDESC27','500 background',after='TUCD27')
    
    tbhdu.header.set('TUCD28','phot.flux.density',after='TUNIT28')
    tbhdu.header.set('TDESC28','250 residual confusion noise',after='TUCD28')

    tbhdu.header.set('TUCD29','phot.flux.density',after='TUNIT29')
    tbhdu.header.set('TDESC29','350 residual confusion noise',after='TUCD29')
    
    tbhdu.header.set('TUCD30','phot.flux.density',after='TUNIT30')
    tbhdu.header.set('TDESC30','500 residual confusion noise',after='TUCD30')

    tbhdu.header.set('TUCD31','phot.redshift',after='TFORM31')      
    tbhdu.header.set('TDESC31','phot.redshift (at 50th percentile)',after='TUCD31') 

    tbhdu.header.set('TUCD32','phot.redshift',after='TFORM32')      
    tbhdu.header.set('TDESC32','phot.redshift (at 84.1 percentile) ',after='TUCD32') 

    tbhdu.header.set('TUCD33','phot.redshift',after='TFORM33')      
    tbhdu.header.set('TDESC33','phot.redshift (at 15.9 percentile)',after='TUCD33')

    tbhdu.header.set('TUCD34','log.Lsol',after='TUNIT34')      
    tbhdu.header.set('TDESC34','log(Luminosity) (at 50th percentile)',after='TUCD34') 

    tbhdu.header.set('TUCD35','log.Lsol',after='TUNIT35')      
    tbhdu.header.set('TDESC35','log(Luminosity) (at 84.1 percentile) ',after='TUCD35') 

    tbhdu.header.set('TUCD36','log.Lsol',after='TUNIT36')      
    tbhdu.header.set('TDESC36','log(Luminosity) (at 15.9 percentile)',after='TUCD36')


    tbhdu.header.set('TUCD37','stat.value',after='TFORM37')
    tbhdu.header.set('TDESC37','250 MCMC Convergence statistic',after='TUCD37')

    tbhdu.header.set('TUCD38','stat.value',after='TFORM38')
    tbhdu.header.set('TDESC38','350 MCMC Convergence statistic',after='TUCD38')

    tbhdu.header.set('TUCD39','stat.value',after='TFORM39')
    tbhdu.header.set('TDESC39','500 MCMC Convergence statistic',after='TUCD39')
    
    tbhdu.header.set('TUCD40','stat.value',after='TFORM40')
    tbhdu.header.set('TDESC40','250 MCMC independence statistic',after='TUCD40')

    tbhdu.header.set('TUCD41','stat.value',after='TFORM41')
    tbhdu.header.set('TDESC41','350 MCMC independence statistic',after='TUCD41')

    tbhdu.header.set('TUCD42','stat.value',after='TFORM42')
    tbhdu.header.set('TDESC42','500 MCMC independence statistic',after='TUCD42')
    
    tbhdu.header.set('TUCD43','stat.value',after='TFORM43')
    tbhdu.header.set('TDESC43','250 Bayes Pval residual statistic',after='TUCD43')

    tbhdu.header.set('TUCD44','stat.value',after='TFORM44')
    tbhdu.header.set('TDESC44','350 Bayes Pval residual statistic',after='TUCD44')

    tbhdu.header.set('TUCD45','stat.value',after='TFORM45')
    tbhdu.header.set('TDESC45','500 Bayes Pval residual statistic',after='TUCD45')
    
    
    

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'XID+SED catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist








def create_Herchel_gaussprior_cat(posterior,priors):

    """
    Create SED catalogue from posterior


    :param posterior: SED xidplus.posterior class
    :param priors: MIPS-PACS-SPIRE-LABOCA xidplus.prior class
    :return: fits hdulist
    """
    
    import datetime
    nsrc=posterior.nsrc
    prior24  = priors[0]
    prior100 = priors[1]
    prior160 = priors[2]
    prior250 = priors[3]
    prior350 = priors[4]
    prior500 = priors[5]
    prior850 = priors[6]
    
    rep_maps = postmaps.replicated_maps([prior100,prior160, prior250, prior350,prior500], posterior)
    Bayes_P100 = postmaps.Bayes_Pval_res(prior350, rep_maps[0])
    Bayes_P160 = postmaps.Bayes_Pval_res(prior500, rep_maps[1])
    Bayes_P250 = postmaps.Bayes_Pval_res(prior250, rep_maps[2])
    Bayes_P350 = postmaps.Bayes_Pval_res(prior350, rep_maps[3])
    Bayes_P500 = postmaps.Bayes_Pval_res(prior500, rep_maps[4])

    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_MIPS_24', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_MIPS_24_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_MIPS_24_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_PACS_100', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_PACS_100_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_PACS_100_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_PACS_160', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_PACS_160_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_PACS_160_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],50.0,axis=0))
    c14 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],84.1,axis=0))
    c15 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],15.9,axis=0))
    c16 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],50.0,axis=0))
    c17 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],84.1,axis=0))
    c18 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],15.9,axis=0))
    c19 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],50.0,axis=0))
    c20 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],84.1,axis=0))
    c21 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],15.9,axis=0))
    c22 = fits.Column(name='F_LABOCA_850', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],50.0,axis=0))
    c23 = fits.Column(name='FErr_LABOCA_850_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],84.1,axis=0))
    c24 = fits.Column(name='FErr_LABOCA_850_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],15.9,axis=0))
    
    
    c25 = fits.Column(name='Bkg_PACS_100', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c26 = fits.Column(name='Bkg_PACS_160', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,1],50.0,axis=0)))
    
    c27 = fits.Column(name='Bkg_SPIRE_250', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,2],50.0,axis=0)))
    c28 = fits.Column(name='Bkg_SPIRE_350', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,3],50.0,axis=0)))
    c29 = fits.Column(name='Bkg_SPIRE_500', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,4],50.0,axis=0)))
    
    
    
    c30 = fits.Column(name='Sig_conf_PACS_100', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c31 = fits.Column(name='Sig_conf_PACS_160', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,1],50.0,axis=0)))
    c32 = fits.Column(name='Sig_conf_SPIRE_250', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,2],50.0,axis=0)))
    c33 = fits.Column(name='Sig_conf_SPIRE_350', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,3],50.0,axis=0)))
    c34 = fits.Column(name='Sig_conf_SPIRE_500', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,4],50.0,axis=0)))
    
    
    c35 = fits.Column(name='z', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],50.0,axis=0))
    c36 = fits.Column(name='zErr_u', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],84.1,axis=0))
    c37 = fits.Column(name='zErr_l', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],15.9,axis=0))
    c38 = fits.Column(name='Nbb', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],50.0,axis=0))
    c39 = fits.Column(name='Nbb_u', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],84.1,axis=0))
    c40 = fits.Column(name='Nbb_l', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],15.9,axis=0))
    
    
    c41 = fits.Column(name='Rhat_MIPS_24', format='E', array=posterior.Rhat['src_f'][:,0])
    c42 = fits.Column(name='n_eff_MIPS_24', format='E', array=posterior.n_eff['src_f'][:,0])


    c43 = fits.Column(name='Rhat_PACS_100', format='E', array=posterior.Rhat['src_f'][:,1])
    c44 = fits.Column(name='Rhat_PACS_160', format='E', array=posterior.Rhat['src_f'][:,2])
    c45 = fits.Column(name='n_eff_PACS_100', format='E', array=posterior.n_eff['src_f'][:,1])
    c46 = fits.Column(name='n_eff_PACS_160', format='E', array=posterior.n_eff['src_f'][:,2])
    c47 = fits.Column(name='Pval_res_100', format='E', array=Bayes_P100)
    c48 = fits.Column(name='Pval_res_160', format='E', array=Bayes_P160)
    
    c49 = fits.Column(name='Rhat_SPIRE_250', format='E', array=posterior.Rhat['src_f'][:,3])
    c50 = fits.Column(name='Rhat_SPIRE_350', format='E', array=posterior.Rhat['src_f'][:,4])
    c51 = fits.Column(name='Rhat_SPIRE_500', format='E', array=posterior.Rhat['src_f'][:,5])
    c52 = fits.Column(name='n_eff_SPIRE_250', format='E', array=posterior.n_eff['src_f'][:,3])
    c53 = fits.Column(name='n_eff_SPIRE_350', format='E', array=posterior.n_eff['src_f'][:,4])
    c54 = fits.Column(name='n_eff_SPIRE_500', format='E', array=posterior.n_eff['src_f'][:,5])
    c55 = fits.Column(name='Pval_res_250', format='E', array=Bayes_P250)
    c56 = fits.Column(name='Pval_res_350', format='E', array=Bayes_P350)
    c57 = fits.Column(name='Pval_res_500', format='E', array=Bayes_P500)
    
    c58 = fits.Column(name='Rhat_LABOCA_850', format='E', array=posterior.Rhat['src_f'][:,6])
    c59 = fits.Column(name='n_eff_LABOCA_850', format='E', array=posterior.n_eff['src_f'][:,6])
    
        

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,
                                           c30,c31,c32,c33,c34,
                                           c35,c36,c37,c38,c39,c40,
                                           c41,c42,c43,c44,c45,c46,c47,c48,c49,c50,
                                           c51,c52,c53,c54,c55,c56,c57,c58,c59])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','24 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','24 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','24 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','100 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','100 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','100 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','160 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','160 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','160 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 Flux (at 50th percentile)',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','250 Flux (at 84.1 percentile) ',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','250 Flux (at 15.9 percentile)',after='TUCD15') 

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')      
    tbhdu.header.set('TDESC16','350 Flux (at 50th percentile)',after='TUCD16') 

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')      
    tbhdu.header.set('TDESC17','350 Flux (at 84.1 percentile) ',after='TUCD17') 

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')      
    tbhdu.header.set('TDESC18','350 Flux (at 15.9 percentile)',after='TUCD18') 

    tbhdu.header.set('TUCD19','phot.flux.density',after='TUNIT19')      
    tbhdu.header.set('TDESC19','500 Flux (at 50th percentile)',after='TUCD19') 

    tbhdu.header.set('TUCD20','phot.flux.density',after='TUNIT20')      
    tbhdu.header.set('TDESC20','500 Flux (at 84.1 percentile) ',after='TUCD20') 

    tbhdu.header.set('TUCD21','phot.flux.density',after='TUNIT21')      
    tbhdu.header.set('TDESC21','500 Flux (at 15.9 percentile)',after='TUCD21')

    tbhdu.header.set('TUCD22','phot.flux.density',after='TUNIT22')      
    tbhdu.header.set('TDESC22','850 Flux (at 50th percentile)',after='TUCD22') 

    tbhdu.header.set('TUCD23','phot.flux.density',after='TUNIT23')      
    tbhdu.header.set('TDESC23','850 Flux (at 84.1 percentile) ',after='TUCD23') 

    tbhdu.header.set('TUCD24','phot.flux.density',after='TUNIT24')      
    tbhdu.header.set('TDESC24','850 Flux (at 15.9 percentile)',after='TUCD24')
    

    tbhdu.header.set('TUCD25','phot.flux.density',after='TUNIT25')      
    tbhdu.header.set('TDESC25','100 background',after='TUCD25') 

    tbhdu.header.set('TUCD26','phot.flux.density',after='TUNIT26')      
    tbhdu.header.set('TDESC26','160 background',after='TUCD26')
    
    tbhdu.header.set('TUCD27','phot.flux.density',after='TUNIT27')      
    tbhdu.header.set('TDESC27','250 background',after='TUCD27') 

    tbhdu.header.set('TUCD28','phot.flux.density',after='TUNIT28')      
    tbhdu.header.set('TDESC28','350 background',after='TUCD28') 

    tbhdu.header.set('TUCD29','phot.flux.density',after='TUNIT29')      
    tbhdu.header.set('TDESC29','500 background',after='TUCD29')

    tbhdu.header.set('TUCD30','phot.flux.density',after='TUNIT30')
    tbhdu.header.set('TDESC30','100 residual confusion noise',after='TUCD30')

    tbhdu.header.set('TUCD31','phot.flux.density',after='TUNIT31')
    tbhdu.header.set('TDESC31','160 residual confusion noise',after='TUCD31')
    
    tbhdu.header.set('TUCD32','phot.flux.density',after='TUNIT32')
    tbhdu.header.set('TDESC32','250 residual confusion noise',after='TUCD32')

    tbhdu.header.set('TUCD33','phot.flux.density',after='TUNIT33')
    tbhdu.header.set('TDESC33','350 residual confusion noise',after='TUCD33')
    
    tbhdu.header.set('TUCD34','phot.flux.density',after='TUNIT34')
    tbhdu.header.set('TDESC34','500 residual confusion noise',after='TUCD34')

    tbhdu.header.set('TUCD35','phot.redshift',after='TFORM35')      
    tbhdu.header.set('TDESC35','phot.redshift (at 50th percentile)',after='TUCD35') 

    tbhdu.header.set('TUCD36','phot.redshift',after='TFORM36')      
    tbhdu.header.set('TDESC36','phot.redshift (at 84.1 percentile) ',after='TUCD36') 

    tbhdu.header.set('TUCD37','phot.redshift',after='TFORM37')      
    tbhdu.header.set('TDESC37','phot.redshift (at 15.9 percentile)',after='TUCD37')

    tbhdu.header.set('TUCD38','log.Lsol',after='TUNIT38')      
    tbhdu.header.set('TDESC38','log(Luminosity) (at 50th percentile)',after='TUCD38') 

    tbhdu.header.set('TUCD39','log.Lsol',after='TUNIT39')      
    tbhdu.header.set('TDESC39','log(Luminosity) (at 84.1 percentile) ',after='TUCD39') 

    tbhdu.header.set('TUCD40','log.Lsol',after='TUNIT40')      
    tbhdu.header.set('TDESC40','log(Luminosity) (at 15.9 percentile)',after='TUCD40')
    
    
    tbhdu.header.set('TUCD41','stat.value',after='TFORM41')
    tbhdu.header.set('TDESC41','24 MCMC Convergence statistic',after='TUCD41')
    
    tbhdu.header.set('TUCD42','stat.value',after='TFORM42')
    tbhdu.header.set('TDESC42','24 MCMC independence statistic',after='TUCD42')
    

    tbhdu.header.set('TUCD43','stat.value',after='TFORM43')
    tbhdu.header.set('TDESC43','100 MCMC Convergence statistic',after='TUCD43')

    tbhdu.header.set('TUCD44','stat.value',after='TFORM44')
    tbhdu.header.set('TDESC44','160 MCMC Convergence statistic',after='TUCD44')
    
    tbhdu.header.set('TUCD45','stat.value',after='TFORM45')
    tbhdu.header.set('TDESC45','100 MCMC independence statistic',after='TUCD45')

    tbhdu.header.set('TUCD46','stat.value',after='TFORM46')
    tbhdu.header.set('TDESC46','160 MCMC independence statistic',after='TUCD46')
    
    tbhdu.header.set('TUCD47','stat.value',after='TFORM47')
    tbhdu.header.set('TDESC47','100 Bayes Pval residual statistic',after='TUCD47')

    tbhdu.header.set('TUCD48','stat.value',after='TFORM48')
    tbhdu.header.set('TDESC48','160 Bayes Pval residual statistic',after='TUCD48')
    

    tbhdu.header.set('TUCD49','stat.value',after='TFORM49')
    tbhdu.header.set('TDESC49','250 MCMC Convergence statistic',after='TUCD49')

    tbhdu.header.set('TUCD50','stat.value',after='TFORM50')
    tbhdu.header.set('TDESC50','350 MCMC Convergence statistic',after='TUCD50')

    tbhdu.header.set('TUCD51','stat.value',after='TFORM51')
    tbhdu.header.set('TDESC51','500 MCMC Convergence statistic',after='TUCD51')
    
    tbhdu.header.set('TUCD52','stat.value',after='TFORM52')
    tbhdu.header.set('TDESC52','250 MCMC independence statistic',after='TUCD52')

    tbhdu.header.set('TUCD53','stat.value',after='TFORM53')
    tbhdu.header.set('TDESC53','350 MCMC independence statistic',after='TUCD53')

    tbhdu.header.set('TUCD54','stat.value',after='TFORM54')
    tbhdu.header.set('TDESC54','500 MCMC independence statistic',after='TUCD54')
    
    tbhdu.header.set('TUCD55','stat.value',after='TFORM55')
    tbhdu.header.set('TDESC55','250 Bayes Pval residual statistic',after='TUCD55')

    tbhdu.header.set('TUCD56','stat.value',after='TFORM56')
    tbhdu.header.set('TDESC56','350 Bayes Pval residual statistic',after='TUCD56')

    tbhdu.header.set('TUCD57','stat.value',after='TFORM57')
    tbhdu.header.set('TDESC57','500 Bayes Pval residual statistic',after='TUCD57')
    
    tbhdu.header.set('TUCD58','stat.value',after='TFORM58')
    tbhdu.header.set('TDESC58','850 MCMC Convergence statistic',after='TUCD58')
    
    tbhdu.header.set('TUCD59','stat.value',after='TFORM59')
    tbhdu.header.set('TDESC59','850 MCMC independence statistic',after='TUCD59')
    
    
    
    

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'XID+SED catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist


def create_IR_gaussprior_cat(posterior,priors):

    """
    Create SED catalogue from posterior


    :param posterior: SED xidplus.posterior class
    :param priors: MIPS-PACS-SPIRE-LABOCA xidplus.prior class
    :return: fits hdulist
    """
    
    import datetime
    nsrc=posterior.nsrc
    prior24  = priors[0]
    prior100 = priors[1]
    prior160 = priors[2]
    prior250 = priors[3]
    prior350 = priors[4]
    prior500 = priors[5]
    prior850 = priors[6]
    
    rep_maps = postmaps.replicated_maps([prior24, prior100,prior160, prior250, prior350,prior500], posterior)
    Bayes_P24 = postmaps.Bayes_Pval_res(prior250, rep_maps[0])
    Bayes_P100 = postmaps.Bayes_Pval_res(prior350, rep_maps[1])
    Bayes_P160 = postmaps.Bayes_Pval_res(prior500, rep_maps[2])
    Bayes_P250 = postmaps.Bayes_Pval_res(prior250, rep_maps[3])
    Bayes_P350 = postmaps.Bayes_Pval_res(prior350, rep_maps[4])
    Bayes_P500 = postmaps.Bayes_Pval_res(prior500, rep_maps[5])

    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_MIPS_24', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_MIPS_24_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_MIPS_24_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_PACS_100', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_PACS_100_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_PACS_100_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_PACS_160', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_PACS_160_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_PACS_160_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],50.0,axis=0))
    c14 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],84.1,axis=0))
    c15 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],15.9,axis=0))
    c16 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],50.0,axis=0))
    c17 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],84.1,axis=0))
    c18 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],15.9,axis=0))
    c19 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],50.0,axis=0))
    c20 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],84.1,axis=0))
    c21 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],15.9,axis=0))
    c22 = fits.Column(name='F_LABOCA_850', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],50.0,axis=0))
    c23 = fits.Column(name='FErr_LABOCA_850_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],84.1,axis=0))
    c24 = fits.Column(name='FErr_LABOCA_850_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],15.9,axis=0))
    
    c25 = fits.Column(name='Bkg_MIPS_24', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c26 = fits.Column(name='Bkg_PACS_100', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,1],50.0,axis=0)))
    c27 = fits.Column(name='Bkg_PACS_160', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,2],50.0,axis=0)))
    
    c28 = fits.Column(name='Bkg_SPIRE_250', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,3],50.0,axis=0)))
    c29 = fits.Column(name='Bkg_SPIRE_350', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,4],50.0,axis=0)))
    c30 = fits.Column(name='Bkg_SPIRE_500', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,5],50.0,axis=0)))
    
    
    c31 = fits.Column(name='Sig_conf_MIPS_24', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c32 = fits.Column(name='Sig_conf_PACS_100', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,1],50.0,axis=0)))
    c33 = fits.Column(name='Sig_conf_PACS_160', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,2],50.0,axis=0)))
    c34 = fits.Column(name='Sig_conf_SPIRE_250', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,3],50.0,axis=0)))
    c35 = fits.Column(name='Sig_conf_SPIRE_350', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,4],50.0,axis=0)))
    c36 = fits.Column(name='Sig_conf_SPIRE_500', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,5],50.0,axis=0)))
    
    
    c37 = fits.Column(name='z', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],50.0,axis=0))
    c38 = fits.Column(name='zErr_u', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],84.1,axis=0))
    c39 = fits.Column(name='zErr_l', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],15.9,axis=0))
    c40 = fits.Column(name='Nbb', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],50.0,axis=0))
    c41 = fits.Column(name='Nbb_u', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],84.1,axis=0))
    c42 = fits.Column(name='Nbb_l', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],15.9,axis=0))
    
    
    c43 = fits.Column(name='Rhat_MIPS_24', format='E', array=posterior.Rhat['src_f'][:,0])
    c44 = fits.Column(name='n_eff_MIPS_24', format='E', array=posterior.n_eff['src_f'][:,0])
    c45 = fits.Column(name='Pval_res_24', format='E', array=Bayes_P24)

    c46 = fits.Column(name='Rhat_PACS_100', format='E', array=posterior.Rhat['src_f'][:,1])
    c47 = fits.Column(name='Rhat_PACS_160', format='E', array=posterior.Rhat['src_f'][:,2])
    c48 = fits.Column(name='n_eff_PACS_100', format='E', array=posterior.n_eff['src_f'][:,1])
    c49 = fits.Column(name='n_eff_PACS_160', format='E', array=posterior.n_eff['src_f'][:,2])
    c50 = fits.Column(name='Pval_res_100', format='E', array=Bayes_P100)
    c51 = fits.Column(name='Pval_res_160', format='E', array=Bayes_P160)
    
    c52 = fits.Column(name='Rhat_SPIRE_250', format='E', array=posterior.Rhat['src_f'][:,3])
    c53 = fits.Column(name='Rhat_SPIRE_350', format='E', array=posterior.Rhat['src_f'][:,4])
    c54 = fits.Column(name='Rhat_SPIRE_500', format='E', array=posterior.Rhat['src_f'][:,5])
    c55 = fits.Column(name='n_eff_SPIRE_250', format='E', array=posterior.n_eff['src_f'][:,3])
    c56 = fits.Column(name='n_eff_SPIRE_350', format='E', array=posterior.n_eff['src_f'][:,4])
    c57 = fits.Column(name='n_eff_SPIRE_500', format='E', array=posterior.n_eff['src_f'][:,5])
    c58 = fits.Column(name='Pval_res_250', format='E', array=Bayes_P250)
    c59 = fits.Column(name='Pval_res_350', format='E', array=Bayes_P350)
    c60 = fits.Column(name='Pval_res_500', format='E', array=Bayes_P500)
    
    c61 = fits.Column(name='Rhat_LABOCA_850', format='E', array=posterior.Rhat['src_f'][:,6])
    c62 = fits.Column(name='n_eff_LABOCA_850', format='E', array=posterior.n_eff['src_f'][:,6])
    
        

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,
                                           c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,
                                           c41,c42,c43,c44,c45,c46,c47,c48,c49,c50,
                                           c51,c52,c53,c54,c55,c56,c57,c58,c59,c60,c61,c62])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','24 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','24 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','24 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','100 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','100 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','100 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','160 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','160 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','160 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 Flux (at 50th percentile)',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','250 Flux (at 84.1 percentile) ',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','250 Flux (at 15.9 percentile)',after='TUCD15') 

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')      
    tbhdu.header.set('TDESC16','350 Flux (at 50th percentile)',after='TUCD16') 

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')      
    tbhdu.header.set('TDESC17','350 Flux (at 84.1 percentile) ',after='TUCD17') 

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')      
    tbhdu.header.set('TDESC18','350 Flux (at 15.9 percentile)',after='TUCD18') 

    tbhdu.header.set('TUCD19','phot.flux.density',after='TUNIT19')      
    tbhdu.header.set('TDESC19','500 Flux (at 50th percentile)',after='TUCD19') 

    tbhdu.header.set('TUCD20','phot.flux.density',after='TUNIT20')      
    tbhdu.header.set('TDESC20','500 Flux (at 84.1 percentile) ',after='TUCD20') 

    tbhdu.header.set('TUCD21','phot.flux.density',after='TUNIT21')      
    tbhdu.header.set('TDESC21','500 Flux (at 15.9 percentile)',after='TUCD21')

    tbhdu.header.set('TUCD22','phot.flux.density',after='TUNIT22')      
    tbhdu.header.set('TDESC22','850 Flux (at 50th percentile)',after='TUCD22') 

    tbhdu.header.set('TUCD23','phot.flux.density',after='TUNIT23')      
    tbhdu.header.set('TDESC23','850 Flux (at 84.1 percentile) ',after='TUCD23') 

    tbhdu.header.set('TUCD24','phot.flux.density',after='TUNIT24')      
    tbhdu.header.set('TDESC24','850 Flux (at 15.9 percentile)',after='TUCD24')
    

    tbhdu.header.set('TUCD25','phot.flux.density',after='TUNIT25')      
    tbhdu.header.set('TDESC25','24 background',after='TUCD25') 

    tbhdu.header.set('TUCD26','phot.flux.density',after='TUNIT26')      
    tbhdu.header.set('TDESC26','100 background',after='TUCD25') 

    tbhdu.header.set('TUCD27','phot.flux.density',after='TUNIT27')      
    tbhdu.header.set('TDESC27','160 background',after='TUCD27')
    
    tbhdu.header.set('TUCD28','phot.flux.density',after='TUNIT28')      
    tbhdu.header.set('TDESC28','250 background',after='TUCD28') 

    tbhdu.header.set('TUCD29','phot.flux.density',after='TUNIT29')      
    tbhdu.header.set('TDESC29','350 background',after='TUCD29') 

    tbhdu.header.set('TUCD30','phot.flux.density',after='TUNIT30')      
    tbhdu.header.set('TDESC30','500 background',after='TUCD30')

    tbhdu.header.set('TUCD31','phot.flux.density',after='TUNIT31')
    tbhdu.header.set('TDESC31','24 residual confusion noise',after='TUCD31')

    tbhdu.header.set('TUCD32','phot.flux.density',after='TUNIT32')
    tbhdu.header.set('TDESC32','100 residual confusion noise',after='TUCD32')

    tbhdu.header.set('TUCD33','phot.flux.density',after='TUNIT33')
    tbhdu.header.set('TDESC33','160 residual confusion noise',after='TUCD33')
    
    tbhdu.header.set('TUCD34','phot.flux.density',after='TUNIT34')
    tbhdu.header.set('TDESC34','250 residual confusion noise',after='TUCD34')

    tbhdu.header.set('TUCD35','phot.flux.density',after='TUNIT35')
    tbhdu.header.set('TDESC35','350 residual confusion noise',after='TUCD35')

    tbhdu.header.set('TUCD36','phot.flux.density',after='TUNIT36')
    tbhdu.header.set('TDESC36','500 residual confusion noise',after='TUCD36')

    tbhdu.header.set('TUCD37','phot.redshift',after='TFORM37')      
    tbhdu.header.set('TDESC37','phot.redshift (at 50th percentile)',after='TUCD37') 

    tbhdu.header.set('TUCD38','phot.redshift',after='TFORM38')      
    tbhdu.header.set('TDESC38','phot.redshift (at 84.1 percentile) ',after='TUCD38') 

    tbhdu.header.set('TUCD39','phot.redshift',after='TFORM39')      
    tbhdu.header.set('TDESC39','phot.redshift (at 15.9 percentile)',after='TUCD39')

    tbhdu.header.set('TUCD40','log.Lsol',after='TUNIT40')      
    tbhdu.header.set('TDESC40','log(Luminosity) (at 50th percentile)',after='TUCD40') 

    tbhdu.header.set('TUCD41','log.Lsol',after='TUNIT41')      
    tbhdu.header.set('TDESC41','log(Luminosity) (at 84.1 percentile) ',after='TUCD41') 

    tbhdu.header.set('TUCD42','log.Lsol',after='TUNIT42')      
    tbhdu.header.set('TDESC42','log(Luminosity) (at 15.9 percentile)',after='TUCD42')
    
    
    tbhdu.header.set('TUCD43','stat.value',after='TFORM43')
    tbhdu.header.set('TDESC43','24 MCMC Convergence statistic',after='TUCD43')
    
    tbhdu.header.set('TUCD44','stat.value',after='TFORM44')
    tbhdu.header.set('TDESC44','24 MCMC independence statistic',after='TUCD44')
    
    tbhdu.header.set('TUCD45','stat.value',after='TFORM45')
    tbhdu.header.set('TDESC45','24 Bayes Pval residual statistic',after='TUCD45')
    

    tbhdu.header.set('TUCD46','stat.value',after='TFORM46')
    tbhdu.header.set('TDESC46','100 MCMC Convergence statistic',after='TUCD46')

    tbhdu.header.set('TUCD47','stat.value',after='TFORM47')
    tbhdu.header.set('TDESC47','160 MCMC Convergence statistic',after='TUCD47')
    
    tbhdu.header.set('TUCD48','stat.value',after='TFORM48')
    tbhdu.header.set('TDESC48','100 MCMC independence statistic',after='TUCD48')

    tbhdu.header.set('TUCD49','stat.value',after='TFORM49')
    tbhdu.header.set('TDESC49','160 MCMC independence statistic',after='TUCD49')
    
    tbhdu.header.set('TUCD50','stat.value',after='TFORM50')
    tbhdu.header.set('TDESC50','100 Bayes Pval residual statistic',after='TUCD50')

    tbhdu.header.set('TUCD51','stat.value',after='TFORM51')
    tbhdu.header.set('TDESC51','160 Bayes Pval residual statistic',after='TUCD51')
    
    
     
    tbhdu.header.set('TUCD52','stat.value',after='TFORM52')
    tbhdu.header.set('TDESC52','250 MCMC Convergence statistic',after='TUCD52')

    tbhdu.header.set('TUCD53','stat.value',after='TFORM53')
    tbhdu.header.set('TDESC53','350 MCMC Convergence statistic',after='TUCD53')

    tbhdu.header.set('TUCD54','stat.value',after='TFORM54')
    tbhdu.header.set('TDESC54','500 MCMC Convergence statistic',after='TUCD54')

    tbhdu.header.set('TUCD55','stat.value',after='TFORM55')
    tbhdu.header.set('TDESC55','250 MCMC independence statistic',after='TUCD55')

    tbhdu.header.set('TUCD56','stat.value',after='TFORM56')
    tbhdu.header.set('TDESC56','350 MCMC independence statistic',after='TUCD56')

    tbhdu.header.set('TUCD57','stat.value',after='TFORM57')
    tbhdu.header.set('TDESC57','500 MCMC independence statistic',after='TUCD57')
    
    tbhdu.header.set('TUCD58','stat.value',after='TFORM58')
    tbhdu.header.set('TDESC58','250 Bayes Pval residual statistic',after='TUCD58')

    tbhdu.header.set('TUCD59','stat.value',after='TFORM59')
    tbhdu.header.set('TDESC59','350 Bayes Pval residual statistic',after='TUCD59')

    tbhdu.header.set('TUCD60','stat.value',after='TFORM60')
    tbhdu.header.set('TDESC60','500 Bayes Pval residual statistic',after='TUCD60')

    tbhdu.header.set('TUCD61','stat.value',after='TFORM61')
    tbhdu.header.set('TDESC61','850 MCMC Convergence statistic',after='TUCD61')
    
    tbhdu.header.set('TUCD62','stat.value',after='TFORM62')
    tbhdu.header.set('TDESC62','850 MCMC independence statistic',after='TUCD62')
    
    

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'XID+SED catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist



def create_SED_cat(posterior,priors):

    """
    Create SED catalogue from posterior


    :param posterior: SED xidplus.posterior class
    :param priors: MIPS-PACS-SPIRE-LABOCA xidplus.prior class
    :return: fits hdulist
    """
    
    import datetime
    nsrc=posterior.nsrc
    prior24  = priors[0]
    prior100 = priors[1]
    prior160 = priors[2]
    prior250 = priors[3]
    prior350 = priors[4]
    prior500 = priors[5]
    prior850 = priors[6]
    
    rep_maps = postmaps.replicated_maps([prior24, prior100,prior160, prior250, prior350,prior500, prior850], posterior)
    Bayes_P24 = postmaps.Bayes_Pval_res(prior250, rep_maps[0])
    Bayes_P100 = postmaps.Bayes_Pval_res(prior350, rep_maps[1])
    Bayes_P160 = postmaps.Bayes_Pval_res(prior500, rep_maps[2])
    Bayes_P250 = postmaps.Bayes_Pval_res(prior250, rep_maps[3])
    Bayes_P350 = postmaps.Bayes_Pval_res(prior350, rep_maps[4])
    Bayes_P500 = postmaps.Bayes_Pval_res(prior500, rep_maps[5])
    Bayes_P850 = postmaps.Bayes_Pval_res(prior500, rep_maps[6])

    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_MIPS_24', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_MIPS_24_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_MIPS_24_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_PACS_100', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_PACS_100_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_PACS_100_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_PACS_160', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_PACS_160_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_PACS_160_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],50.0,axis=0))
    c14 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],84.1,axis=0))
    c15 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],15.9,axis=0))
    c16 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],50.0,axis=0))
    c17 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],84.1,axis=0))
    c18 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],15.9,axis=0))
    c19 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],50.0,axis=0))
    c20 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],84.1,axis=0))
    c21 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],15.9,axis=0))
    c22 = fits.Column(name='F_LABOCA_850', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],50.0,axis=0))
    c23 = fits.Column(name='FErr_LABOCA_850_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],84.1,axis=0))
    c24 = fits.Column(name='FErr_LABOCA_850_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],15.9,axis=0))
    
    c25 = fits.Column(name='Bkg_MIPS_24', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c26 = fits.Column(name='Bkg_PACS_100', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,1],50.0,axis=0)))
    c27 = fits.Column(name='Bkg_PACS_160', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,2],50.0,axis=0)))
    
    c28 = fits.Column(name='Bkg_SPIRE_250', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,3],50.0,axis=0)))
    c29 = fits.Column(name='Bkg_SPIRE_350', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,4],50.0,axis=0)))
    c30 = fits.Column(name='Bkg_SPIRE_500', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,5],50.0,axis=0)))
    c31 = fits.Column(name='Bkg_LABOCA_850', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,6],50.0,axis=0)))
    
    
    c32 = fits.Column(name='Sig_conf_MIPS_24', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c33 = fits.Column(name='Sig_conf_PACS_100', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,1],50.0,axis=0)))
    c34 = fits.Column(name='Sig_conf_PACS_160', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,2],50.0,axis=0)))
    c35 = fits.Column(name='Sig_conf_SPIRE_250', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,3],50.0,axis=0)))
    c36 = fits.Column(name='Sig_conf_SPIRE_350', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,4],50.0,axis=0)))
    c37 = fits.Column(name='Sig_conf_SPIRE_500', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,5],50.0,axis=0)))
    c38 = fits.Column(name='Sig_conf_LABOCA_850', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,6],50.0,axis=0)))
    
    
    c39 = fits.Column(name='z', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],50.0,axis=0))
    c40 = fits.Column(name='zErr_u', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],84.1,axis=0))
    c41 = fits.Column(name='zErr_l', format='E',
                      array=np.percentile(posterior.samples['z'][:,:],15.9,axis=0))
    c42 = fits.Column(name='Nbb', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],50.0,axis=0))
    c43 = fits.Column(name='Nbb_u', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],84.1,axis=0))
    c44 = fits.Column(name='Nbb_l', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],15.9,axis=0))
    
    
    c45 = fits.Column(name='Rhat_MIPS_24', format='E', array=posterior.Rhat['src_f'][:,0])
    c46 = fits.Column(name='n_eff_MIPS_24', format='E', array=posterior.n_eff['src_f'][:,0])
    c47 = fits.Column(name='Pval_res_24', format='E', array=Bayes_P24)

    c48 = fits.Column(name='Rhat_PACS_100', format='E', array=posterior.Rhat['src_f'][:,1])
    c49 = fits.Column(name='Rhat_PACS_160', format='E', array=posterior.Rhat['src_f'][:,2])
    c50 = fits.Column(name='n_eff_PACS_100', format='E', array=posterior.n_eff['src_f'][:,1])
    c51 = fits.Column(name='n_eff_PACS_160', format='E', array=posterior.n_eff['src_f'][:,2])
    c52 = fits.Column(name='Pval_res_100', format='E', array=Bayes_P100)
    c53 = fits.Column(name='Pval_res_160', format='E', array=Bayes_P160)
    
    c54 = fits.Column(name='Rhat_SPIRE_250', format='E', array=posterior.Rhat['src_f'][:,3])
    c55 = fits.Column(name='Rhat_SPIRE_350', format='E', array=posterior.Rhat['src_f'][:,4])
    c56 = fits.Column(name='Rhat_SPIRE_500', format='E', array=posterior.Rhat['src_f'][:,5])
    c57 = fits.Column(name='n_eff_SPIRE_250', format='E', array=posterior.n_eff['src_f'][:,3])
    c58 = fits.Column(name='n_eff_SPIRE_350', format='E', array=posterior.n_eff['src_f'][:,4])
    c59 = fits.Column(name='n_eff_SPIRE_500', format='E', array=posterior.n_eff['src_f'][:,5])
    c60 = fits.Column(name='Pval_res_250', format='E', array=Bayes_P250)
    c61 = fits.Column(name='Pval_res_350', format='E', array=Bayes_P350)
    c62 = fits.Column(name='Pval_res_500', format='E', array=Bayes_P500)
    
    c63 = fits.Column(name='Rhat_LABOCA_850', format='E', array=posterior.Rhat['src_f'][:,6])
    c64 = fits.Column(name='n_eff_LABOCA_850', format='E', array=posterior.n_eff['src_f'][:,6])
    c65 = fits.Column(name='Pval_res_850', format='E', array=Bayes_P24)

    

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,
                                           c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,
                                           c41,c42,c43,c44,c45,c46,c47,c48,c49,c50,
                                           c51,c52,c53,c54,c55,c56,c57,c58,c59,c60,c61,c62,c63,c64,c65])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','24 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','24 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','24 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','100 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','100 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','100 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','160 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','160 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','160 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 Flux (at 50th percentile)',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','250 Flux (at 84.1 percentile) ',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','250 Flux (at 15.9 percentile)',after='TUCD15') 

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')      
    tbhdu.header.set('TDESC16','350 Flux (at 50th percentile)',after='TUCD16') 

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')      
    tbhdu.header.set('TDESC17','350 Flux (at 84.1 percentile) ',after='TUCD17') 

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')      
    tbhdu.header.set('TDESC18','350 Flux (at 15.9 percentile)',after='TUCD18') 

    tbhdu.header.set('TUCD19','phot.flux.density',after='TUNIT19')      
    tbhdu.header.set('TDESC19','500 Flux (at 50th percentile)',after='TUCD19') 

    tbhdu.header.set('TUCD20','phot.flux.density',after='TUNIT20')      
    tbhdu.header.set('TDESC20','500 Flux (at 84.1 percentile) ',after='TUCD20') 

    tbhdu.header.set('TUCD21','phot.flux.density',after='TUNIT21')      
    tbhdu.header.set('TDESC21','500 Flux (at 15.9 percentile)',after='TUCD21')

    tbhdu.header.set('TUCD22','phot.flux.density',after='TUNIT22')      
    tbhdu.header.set('TDESC22','850 Flux (at 50th percentile)',after='TUCD22') 

    tbhdu.header.set('TUCD23','phot.flux.density',after='TUNIT23')      
    tbhdu.header.set('TDESC23','850 Flux (at 84.1 percentile) ',after='TUCD23') 

    tbhdu.header.set('TUCD24','phot.flux.density',after='TUNIT24')      
    tbhdu.header.set('TDESC24','850 Flux (at 15.9 percentile)',after='TUCD24')
    

    tbhdu.header.set('TUCD25','phot.flux.density',after='TUNIT25')      
    tbhdu.header.set('TDESC25','24 background',after='TUCD25') 

    tbhdu.header.set('TUCD25','phot.flux.density',after='TUNIT26')      
    tbhdu.header.set('TDESC26','100 background',after='TUCD25') 

    tbhdu.header.set('TUCD27','phot.flux.density',after='TUNIT27')      
    tbhdu.header.set('TDESC27','160 background',after='TUCD27')
    
    tbhdu.header.set('TUCD28','phot.flux.density',after='TUNIT28')      
    tbhdu.header.set('TDESC28','250 background',after='TUCD28') 

    tbhdu.header.set('TUCD29','phot.flux.density',after='TUNIT29')      
    tbhdu.header.set('TDESC14','350 background',after='TUCD29') 

    tbhdu.header.set('TUCD30','phot.flux.density',after='TUNIT30')      
    tbhdu.header.set('TDESC30','500 background',after='TUCD30')

    tbhdu.header.set('TUCD31','phot.flux.density',after='TUNIT31')      
    tbhdu.header.set('TDESC31','850 background',after='TUCD31')
    

    tbhdu.header.set('TUCD32','phot.flux.density',after='TUNIT32')
    tbhdu.header.set('TDESC32','24 residual confusion noise',after='TUCD32')

    tbhdu.header.set('TUCD33','phot.flux.density',after='TUNIT33')
    tbhdu.header.set('TDESC33','100 residual confusion noise',after='TUCD33')

    tbhdu.header.set('TUCD34','phot.flux.density',after='TUNIT34')
    tbhdu.header.set('TDESC34','160 residual confusion noise',after='TUCD34')
    
    tbhdu.header.set('TUCD35','phot.flux.density',after='TUNIT35')
    tbhdu.header.set('TDESC35','250 residual confusion noise',after='TUCD35')

    tbhdu.header.set('TUCD36','phot.flux.density',after='TUNIT36')
    tbhdu.header.set('TDESC36','350 residual confusion noise',after='TUCD36')

    tbhdu.header.set('TUCD37','phot.flux.density',after='TUNIT37')
    tbhdu.header.set('TDESC37','500 residual confusion noise',after='TUCD37')

    tbhdu.header.set('TUCD38','phot.flux.density',after='TUNIT38')
    tbhdu.header.set('TDESC38','850 residual confusion noise',after='TUCD38')
    

    tbhdu.header.set('TUCD39','phot.redshift',after='TFORM39')      
    tbhdu.header.set('TDESC39','phot.redshift (at 50th percentile)',after='TUCD39') 

    tbhdu.header.set('TUCD40','phot.redshift',after='TFORM40')      
    tbhdu.header.set('TDESC40','phot.redshift (at 84.1 percentile) ',after='TUCD40') 

    tbhdu.header.set('TUCD41','phot.redshift',after='TFORM41')      
    tbhdu.header.set('TDESC41','phot.redshift (at 15.9 percentile)',after='TUCD41')

    tbhdu.header.set('TUCD42','log.Lsol',after='TUNIT42')      
    tbhdu.header.set('TDESC42','log(Luminosity) (at 50th percentile)',after='TUCD42') 

    tbhdu.header.set('TUCD43','log.Lsol',after='TUNIT43')      
    tbhdu.header.set('TDESC43','log(Luminosity) (at 84.1 percentile) ',after='TUCD43') 

    tbhdu.header.set('TUCD44','log.Lsol',after='TUNIT44')      
    tbhdu.header.set('TDESC44','log(Luminosity) (at 15.9 percentile)',after='TUCD44')
    
    
    tbhdu.header.set('TUCD45','stat.value',after='TFORM45')
    tbhdu.header.set('TDESC45','24 MCMC Convergence statistic',after='TUCD45')
    
    tbhdu.header.set('TUCD46','stat.value',after='TFORM46')
    tbhdu.header.set('TDESC46','24 MCMC independence statistic',after='TUCD46')
    
    tbhdu.header.set('TUCD47','stat.value',after='TFORM47')
    tbhdu.header.set('TDESC47','24 Bayes Pval residual statistic',after='TUCD47')
    

    tbhdu.header.set('TUCD48','stat.value',after='TFORM48')
    tbhdu.header.set('TDESC48','100 MCMC Convergence statistic',after='TUCD48')

    tbhdu.header.set('TUCD49','stat.value',after='TFORM49')
    tbhdu.header.set('TDESC49','160 MCMC Convergence statistic',after='TUCD49')
    
    tbhdu.header.set('TUCD50','stat.value',after='TFORM50')
    tbhdu.header.set('TDESC50','100 MCMC independence statistic',after='TUCD50')

    tbhdu.header.set('TUCD51','stat.value',after='TFORM51')
    tbhdu.header.set('TDESC51','160 MCMC independence statistic',after='TUCD51')
    
    tbhdu.header.set('TUCD52','stat.value',after='TFORM52')
    tbhdu.header.set('TDESC52','100 Bayes Pval residual statistic',after='TUCD52')

    tbhdu.header.set('TUCD53','stat.value',after='TFORM53')
    tbhdu.header.set('TDESC53','160 Bayes Pval residual statistic',after='TUCD53')
    
    
     
    tbhdu.header.set('TUCD54','stat.value',after='TFORM54')
    tbhdu.header.set('TDESC54','250 MCMC Convergence statistic',after='TUCD54')

    tbhdu.header.set('TUCD55','stat.value',after='TFORM55')
    tbhdu.header.set('TDESC55','350 MCMC Convergence statistic',after='TUCD55')

    tbhdu.header.set('TUCD56','stat.value',after='TFORM56')
    tbhdu.header.set('TDESC56','500 MCMC Convergence statistic',after='TUCD56')

    tbhdu.header.set('TUCD57','stat.value',after='TFORM57')
    tbhdu.header.set('TDESC57','250 MCMC independence statistic',after='TUCD57')

    tbhdu.header.set('TUCD58','stat.value',after='TFORM58')
    tbhdu.header.set('TDESC58','350 MCMC independence statistic',after='TUCD58')

    tbhdu.header.set('TUCD59','stat.value',after='TFORM59')
    tbhdu.header.set('TDESC59','500 MCMC independence statistic',after='TUCD59')
    
    tbhdu.header.set('TUCD60','stat.value',after='TFORM60')
    tbhdu.header.set('TDESC60','250 Bayes Pval residual statistic',after='TUCD60')

    tbhdu.header.set('TUCD61','stat.value',after='TFORM61')
    tbhdu.header.set('TDESC61','350 Bayes Pval residual statistic',after='TUCD61')

    tbhdu.header.set('TUCD62','stat.value',after='TFORM62')
    tbhdu.header.set('TDESC62','500 Bayes Pval residual statistic',after='TUCD62')
    

    tbhdu.header.set('TUCD63','stat.value',after='TFORM63')
    tbhdu.header.set('TDESC63','850 MCMC Convergence statistic',after='TUCD63')
    
    tbhdu.header.set('TUCD64','stat.value',after='TFORM64')
    tbhdu.header.set('TDESC64','850 MCMC independence statistic',after='TUCD64')
    
    tbhdu.header.set('TUCD65','stat.value',after='TFORM65')
    tbhdu.header.set('TDESC65','850 Bayes Pval residual statistic',after='TUCD65')

    
    

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'XID+SED catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist


def create_SED_cat_fixz(posterior,priors):

    """
    Create SED catalogue from posterior


    :param posterior: SED xidplus.posterior class
    :param priors: MIPS-PACS-SPIRE-LABOCA xidplus.prior class
    :return: fits hdulist
    """
    import datetime
    nsrc=posterior.nsrc
    prior24  = priors[0]
    prior100 = priors[1]
    prior160 = priors[2]
    prior250 = priors[3]
    prior350 = priors[4]
    prior500 = priors[5]
    prior850 = priors[6]

    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_MIPS_24', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_MIPS_24_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_MIPS_24_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_PACS_100', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_PACS_100_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_PACS_100_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_PACS_160', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_PACS_160_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_PACS_160_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],50.0,axis=0))
    c14 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],84.1,axis=0))
    c15 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,3,:],15.9,axis=0))
    c16 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],50.0,axis=0))
    c17 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],84.1,axis=0))
    c18 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,4,:],15.9,axis=0))
    c19 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],50.0,axis=0))
    c20 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],84.1,axis=0))
    c21 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,5,:],15.9,axis=0))
    c22 = fits.Column(name='F_LABOCA_850', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],50.0,axis=0))
    c23 = fits.Column(name='FErr_LABOCA_850_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],84.1,axis=0))
    c24 = fits.Column(name='FErr_LABOCA_850_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,6,:],15.9,axis=0))
    c25 = fits.Column(name='Nbb', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],50.0,axis=0))
    c26 = fits.Column(name='Nbb_u', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],84.1,axis=0))
    c27 = fits.Column(name='Nbb_l', format='E', unit='logLsol',
                      array=np.percentile(posterior.samples['Nbb'][:,:],15.9,axis=0))
    

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12, c13, c14, c15, c16, c17, c18, c19, c20, c21,
                            c22, c23, c24, c25, c26, c27])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','24 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','24 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','24 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','100 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','100 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','100 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','160 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','160 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','160 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 Flux (at 50th percentile)',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','250 Flux (at 84.1 percentile) ',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','250 Flux (at 15.9 percentile)',after='TUCD15') 

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')      
    tbhdu.header.set('TDESC16','350 Flux (at 50th percentile)',after='TUCD16') 

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')      
    tbhdu.header.set('TDESC17','350 Flux (at 84.1 percentile) ',after='TUCD17') 

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')      
    tbhdu.header.set('TDESC18','350 Flux (at 15.9 percentile)',after='TUCD18') 

    tbhdu.header.set('TUCD19','phot.flux.density',after='TUNIT19')      
    tbhdu.header.set('TDESC19','500 Flux (at 50th percentile)',after='TUCD19') 

    tbhdu.header.set('TUCD20','phot.flux.density',after='TUNIT20')      
    tbhdu.header.set('TDESC20','500 Flux (at 84.1 percentile) ',after='TUCD20') 

    tbhdu.header.set('TUCD21','phot.flux.density',after='TUNIT21')      
    tbhdu.header.set('TDESC21','500 Flux (at 15.9 percentile)',after='TUCD21')

    tbhdu.header.set('TUCD22','phot.flux.density',after='TUNIT22')      
    tbhdu.header.set('TDESC22','850 Flux (at 50th percentile)',after='TUCD22') 

    tbhdu.header.set('TUCD23','phot.flux.density',after='TUNIT23')      
    tbhdu.header.set('TDESC23','850 Flux (at 84.1 percentile) ',after='TUCD23') 

    tbhdu.header.set('TUCD24','phot.flux.density',after='TUNIT24')      
    tbhdu.header.set('TDESC24','850 Flux (at 15.9 percentile)',after='TUCD24')

    tbhdu.header.set('TUCD25','log.Lsol',after='TUNIT25')      
    tbhdu.header.set('TDESC25','log(Luminosity) (at 50th percentile)',after='TUCD25') 

    tbhdu.header.set('TUCD26','log.Lsol',after='TUNIT26')      
    tbhdu.header.set('TDESC26','log(Luminosity) (at 84.1 percentile) ',after='TUCD26') 

    tbhdu.header.set('TUCD27','log.Lsol',after='TUNIT27')      
    tbhdu.header.set('TDESC27','log(Luminosity) (at 15.9 percentile)',after='TUCD27')

    
    

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'XID+SED catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist

