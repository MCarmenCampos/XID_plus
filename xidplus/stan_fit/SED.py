from numpy import long

__author__ = 'pdh21'
import os
output_dir=os.getcwd()
import pystan
import pickle
import inspect
full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)
import numpy as np

stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'

def MIPS_PACS_SPIRE(phot_priors,sed_prior_model,chains=4,iter=1000):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior24=phot_priors[0]
    prior100=phot_priors[1]
    prior160=phot_priors[2]
    prior250=phot_priors[3]
    prior350=phot_priors[4]
    prior500=phot_priors[5]

    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior24.bkg[0], prior100.bkg[0],
                      prior160.bkg[0],prior250.bkg[0], prior350.bkg[0], prior500.bkg[0]],
        'bkg_prior_sig': [prior24.bkg[1], prior100.bkg[1],
                          prior160.bkg[1],prior250.bkg[1], prior350.bkg[1], prior500.bkg[1]],
        'conf_prior_sig': [0.1,0.1, 0.1, 0.1, 0.1, 0.1],
        'z_median': prior24.z_median,
        'z_sig': prior24.z_sig,
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        'npix_mips24': prior24.snpix,
        'nnz_mips24': prior24.amat_data.size,
        'db_mips24': prior24.sim,
        'sigma_mips24': prior24.snim,
        'Val_mips24': prior24.amat_data,
        'Row_mips24': prior24.amat_row.astype(np.long),
        'Col_mips24': prior24.amat_col.astype(np.long),
        'npix_pacs100': prior100.snpix,
        'nnz_pacs100': prior100.amat_data.size,
        'db_pacs100': prior100.sim,
        'sigma_pacs100': prior100.snim,
        'Val_pacs100': prior100.amat_data,
        'Row_pacs100': prior100.amat_row.astype(np.long),
        'Col_pacs100': prior100.amat_col.astype(np.long),
        'npix_pacs160': prior160.snpix,
        'nnz_pacs160': prior160.amat_data.size,
        'db_pacs160': prior160.sim,
        'sigma_pacs160': prior160.snim,
        'Val_pacs160': prior160.amat_data,
        'Row_pacs160': prior160.amat_row.astype(np.long),
        'Col_pacs160': prior160.amat_col.astype(np.long),
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }


    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+IR_SED'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit


def MIPS_PACS_SPIRE_LABOCA(phot_priors,sed_prior_model,chains=4,iter=1000, control=True):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500, LABOCA850)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior24=phot_priors[0]
    prior100=phot_priors[1]
    prior160=phot_priors[2]
    prior250=phot_priors[3]
    prior350=phot_priors[4]
    prior500=phot_priors[5]
    prior850=phot_priors[6]

    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior24.bkg[0], prior100.bkg[0],
                      prior160.bkg[0],prior250.bkg[0], prior350.bkg[0], prior500.bkg[0], prior850.bkg[0]],
        'bkg_prior_sig': [prior24.bkg[1], prior100.bkg[1],
                          prior160.bkg[1],prior250.bkg[1], prior350.bkg[1], prior500.bkg[1], prior850.bkg[1]],
        'conf_prior_sig': [0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'f_low_lim':[prior24.prior_flux_lower, prior100.prior_flux_lower, prior160.prior_flux_lower,
                     prior250.prior_flux_lower, prior350.prior_flux_lower, prior500.prior_flux_lower,
                     prior850.prior_flux_lower],
        'f_up_lim':[prior24.prior_flux_upper,prior100.prior_flux_upper,prior160.prior_flux_upper, 
                    prior250.prior_flux_upper,prior350.prior_flux_upper,prior500.prior_flux_upper, 
                   prior850.prior_flux_upper],
        'z_median': prior24.z_median,
        'z_sig': prior24.z_sig,
        # SPIRE
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        #MIPS
        'npix_mips24': prior24.snpix,
        'nnz_mips24': prior24.amat_data.size,
        'db_mips24': prior24.sim,
        'sigma_mips24': prior24.snim,
        'Val_mips24': prior24.amat_data,
        'Row_mips24': prior24.amat_row.astype(np.long),
        'Col_mips24': prior24.amat_col.astype(np.long),
        #PACS
        'npix_pacs100': prior100.snpix,
        'nnz_pacs100': prior100.amat_data.size,
        'db_pacs100': prior100.sim,
        'sigma_pacs100': prior100.snim,
        'Val_pacs100': prior100.amat_data,
        'Row_pacs100': prior100.amat_row.astype(np.long),
        'Col_pacs100': prior100.amat_col.astype(np.long),
        'npix_pacs160': prior160.snpix,
        'nnz_pacs160': prior160.amat_data.size,
        'db_pacs160': prior160.sim,
        'sigma_pacs160': prior160.snim,
        'Val_pacs160': prior160.amat_data,
        'Row_pacs160': prior160.amat_row.astype(np.long),
        'Col_pacs160': prior160.amat_col.astype(np.long),
        #LABOCA
        'npix_lb850': prior850.snpix,
        'nnz_lb850': prior850.amat_data.size,
        'db_lb850': prior850.sim,
        'sigma_lb850': prior850.snim,
        'Val_lb850': prior850.amat_data,
        'Row_lb850': prior850.amat_row.astype(np.long),
        'Col_lb850': prior850.amat_col.astype(np.long),

        
        #SEDs
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }


    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+IR_Submm_SED'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)
    if control == True:
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    else:
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True, control = {"max_treedepth":12, "adapt_delta": 0.9})
        
    #return fit data
    return fit


def SPIRE_SED(phot_priors,sed_prior_model,chains=4,iter=1000):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior250=phot_priors[0]
    prior350=phot_priors[1]
    prior500=phot_priors[2]
     
    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior250.bkg[0], prior350.bkg[0], prior500.bkg[0]],
        'bkg_prior_sig': [prior250.bkg[1], prior350.bkg[1], prior500.bkg[1]],
        'conf_prior_sig': [0.1, 0.1, 0.1],
        'f_low_lim':[prior250.prior_flux_lower, prior350.prior_flux_lower, prior500.prior_flux_lower],
        'f_up_lim':[prior250.prior_flux_upper,prior350.prior_flux_upper,prior500.prior_flux_upper],
        'z_median': prior250.z_median,
        'z_sig': prior250.z_sig,
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }

    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+SPIRE_SED_flux-limits'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit


def SPIRE_GaussPrior(phot_priors,sed_prior_model,chains=4,iter=1000):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior24=phot_priors[0]
    prior100=phot_priors[1]
    prior160=phot_priors[2]
    prior250=phot_priors[3]
    prior350=phot_priors[4]
    prior500=phot_priors[5]
    prior850=phot_priors[6]
     
# # To use when using gauss_priors for all bands 
#     f_mu = []
#     f_sigma = []
    
#     for p in phot_priors:
#         f_mu.append((p.prior_mean_flux - p.prior_flux_lower) / (p.prior_flux_upper - p.prior_flux_lower))
#         f_sigma.append(p.prior_sigma_flux / (p.prior_flux_upper - p.prior_flux_lower)) 

    
    f_mu_24 = (prior24.prior_mean_flux - prior24.prior_flux_lower) / (prior24.prior_flux_upper - prior24.prior_flux_lower)
    f_sigma_24 = prior24.prior_sigma_flux / (prior24.prior_flux_upper - prior24.prior_flux_lower)
    
    f_mu_100 = (prior100.prior_mean_flux - prior100.prior_flux_lower) / (prior100.prior_flux_upper - prior24.prior_flux_lower)
    f_sigma_100 = prior100.prior_sigma_flux / (prior100.prior_flux_upper - prior100.prior_flux_lower)
    
    f_mu_160 = (prior160.prior_mean_flux - prior160.prior_flux_lower) / (prior160.prior_flux_upper - prior24.prior_flux_lower)
    f_sigma_160 = prior160.prior_sigma_flux / (prior160.prior_flux_upper - prior160.prior_flux_lower)

    f_mu_850 = (prior850.prior_mean_flux - prior850.prior_flux_lower) / (prior850.prior_flux_upper - prior850.prior_flux_lower)
    f_sigma_850 = prior850.prior_sigma_flux / (prior850.prior_flux_upper - prior850.prior_flux_lower)

    
    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior250.bkg[0], prior350.bkg[0], prior500.bkg[0]],
        'bkg_prior_sig': [prior250.bkg[1], prior350.bkg[1], prior500.bkg[1]],
        'conf_prior_sig': [0.1, 0.1, 0.1],
        'f_low_lim':[prior24.prior_flux_lower, prior100.prior_flux_lower, prior160.prior_flux_lower,
                     prior250.prior_flux_lower, prior350.prior_flux_lower, prior500.prior_flux_lower,
                     prior850.prior_flux_lower],
        'f_up_lim':[prior24.prior_flux_upper,prior100.prior_flux_upper,prior160.prior_flux_upper, 
                    prior250.prior_flux_upper,prior350.prior_flux_upper,prior500.prior_flux_upper, 
                   prior850.prior_flux_upper],
        'f_mu': [f_mu_24, f_mu_100, f_mu_160, f_mu_850], #f_mu
        'f_sigma': [f_sigma_24, f_sigma_100, f_sigma_160, f_sigma_850], #f_sigma        
        'z_median': prior250.z_median,
        'z_sig': prior250.z_sig,
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }


    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+SPIRE_SED_GaussPrior'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit



def Herschel_GaussPrior(phot_priors,sed_prior_model,chains=4,iter=1000):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior24=phot_priors[0]
    prior100=phot_priors[1]
    prior160=phot_priors[2]
    prior250=phot_priors[3]
    prior350=phot_priors[4]
    prior500=phot_priors[5]
    prior850=phot_priors[6]
     
# # To use when using gauss_priors for all bands 
#     f_mu = []
#     f_sigma = []
    
#     for p in phot_priors:
#         f_mu.append((p.prior_mean_flux - p.prior_flux_lower) / (p.prior_flux_upper - p.prior_flux_lower))
#         f_sigma.append(p.prior_sigma_flux / (p.prior_flux_upper - p.prior_flux_lower)) 

    
    f_mu_24 = (prior24.prior_mean_flux - prior24.prior_flux_lower) / (prior24.prior_flux_upper - prior24.prior_flux_lower)
    f_sigma_24 = prior24.prior_sigma_flux / (prior24.prior_flux_upper - prior24.prior_flux_lower)

    f_mu_850 = (prior850.prior_mean_flux - prior850.prior_flux_lower) / (prior850.prior_flux_upper - prior850.prior_flux_lower)
    f_sigma_850 = prior850.prior_sigma_flux / (prior850.prior_flux_upper - prior850.prior_flux_lower)

    
    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior100.bkg[0],
                      prior160.bkg[0],prior250.bkg[0], prior350.bkg[0], prior500.bkg[0]],
        'bkg_prior_sig': [prior100.bkg[1],
                          prior160.bkg[1],prior250.bkg[1], prior350.bkg[1], prior500.bkg[1]],
        'conf_prior_sig': [0.1, 0.1, 0.1, 0.1, 0.1],
        'f_low_lim':[prior24.prior_flux_lower, prior100.prior_flux_lower, prior160.prior_flux_lower,
                     prior250.prior_flux_lower, prior350.prior_flux_lower, prior500.prior_flux_lower,
                     prior850.prior_flux_lower],
        'f_up_lim':[prior24.prior_flux_upper,prior100.prior_flux_upper,prior160.prior_flux_upper, 
                    prior250.prior_flux_upper,prior350.prior_flux_upper,prior500.prior_flux_upper, 
                   prior850.prior_flux_upper],
        'f_mu': [f_mu_24, f_mu_850], #f_mu
        'f_sigma': [f_sigma_24, f_sigma_850], #f_sigma        
        'z_median': prior250.z_median,
        'z_sig': prior250.z_sig,
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        'npix_pacs100': prior100.snpix,
        'nnz_pacs100': prior100.amat_data.size,
        'db_pacs100': prior100.sim,
        'sigma_pacs100': prior100.snim,
        'Val_pacs100': prior100.amat_data,
        'Row_pacs100': prior100.amat_row.astype(np.long),
        'Col_pacs100': prior100.amat_col.astype(np.long),
        'npix_pacs160': prior160.snpix,
        'nnz_pacs160': prior160.amat_data.size,
        'db_pacs160': prior160.sim,
        'sigma_pacs160': prior160.snim,
        'Val_pacs160': prior160.amat_data,
        'Row_pacs160': prior160.amat_row.astype(np.long),
        'Col_pacs160': prior160.amat_col.astype(np.long),
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }


    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+Herschel_SED_GaussPrior'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit

def MIPS_PACS_SPIRE_GaussPrior(phot_priors,sed_prior_model,chains=4,iter=1000):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior24=phot_priors[0]
    prior100=phot_priors[1]
    prior160=phot_priors[2]
    prior250=phot_priors[3]
    prior350=phot_priors[4]
    prior500=phot_priors[5]
    prior850=phot_priors[6]
    
    f_mu_24 = (prior24.prior_mean_flux - prior24.prior_flux_lower) / (prior24.prior_flux_upper - prior24.prior_flux_lower)
    f_sigma_24 = prior24.prior_sigma_flux / (prior24.prior_flux_upper - prior24.prior_flux_lower)

    f_mu_850 = (prior850.prior_mean_flux - prior850.prior_flux_lower) / (prior850.prior_flux_upper - prior850.prior_flux_lower)
    f_sigma_850 = prior850.prior_sigma_flux / (prior850.prior_flux_upper - prior850.prior_flux_lower)

    
    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior24.bkg[0], prior100.bkg[0],
                      prior160.bkg[0],prior250.bkg[0], prior350.bkg[0], prior500.bkg[0]],
        'bkg_prior_sig': [prior24.bkg[1], prior100.bkg[1],
                          prior160.bkg[1],prior250.bkg[1], prior350.bkg[1], prior500.bkg[1]],
        'conf_prior_sig': [0.1,0.1, 0.1, 0.1, 0.1, 0.1],
        'f_low_lim':[prior24.prior_flux_lower, prior100.prior_flux_lower, prior160.prior_flux_lower,
                     prior250.prior_flux_lower, prior350.prior_flux_lower, prior500.prior_flux_lower,
                     prior850.prior_flux_lower],
        'f_up_lim':[prior24.prior_flux_upper,prior100.prior_flux_upper,prior160.prior_flux_upper, 
                    prior250.prior_flux_upper,prior350.prior_flux_upper,prior500.prior_flux_upper, 
                   prior850.prior_flux_upper],
        'f_mu': [f_mu_24, f_mu_850], #f_mu
        'f_sigma': [f_sigma_24, f_sigma_850], #f_sigma
        'z_median': prior24.z_median,
        'z_sig': prior24.z_sig,
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        'npix_mips24': prior24.snpix,
        'nnz_mips24': prior24.amat_data.size,
        'db_mips24': prior24.sim,
        'sigma_mips24': prior24.snim,
        'Val_mips24': prior24.amat_data,
        'Row_mips24': prior24.amat_row.astype(np.long),
        'Col_mips24': prior24.amat_col.astype(np.long),
        'npix_pacs100': prior100.snpix,
        'nnz_pacs100': prior100.amat_data.size,
        'db_pacs100': prior100.sim,
        'sigma_pacs100': prior100.snim,
        'Val_pacs100': prior100.amat_data,
        'Row_pacs100': prior100.amat_row.astype(np.long),
        'Col_pacs100': prior100.amat_col.astype(np.long),
        'npix_pacs160': prior160.snpix,
        'nnz_pacs160': prior160.amat_data.size,
        'db_pacs160': prior160.sim,
        'sigma_pacs160': prior160.snim,
        'Val_pacs160': prior160.amat_data,
        'Row_pacs160': prior160.amat_row.astype(np.long),
        'Col_pacs160': prior160.amat_col.astype(np.long),
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }


    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+IR_SED_GaussPrior'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit

def MIPS_PACS_SPIRE_LABOCA_GaussPrior(phot_priors,sed_prior_model,chains=4,iter=1000, control=True):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500, LABOCA850)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior24=phot_priors[0]
    prior100=phot_priors[1]
    prior160=phot_priors[2]
    prior250=phot_priors[3]
    prior350=phot_priors[4]
    prior500=phot_priors[5]
    prior850=phot_priors[6]

    
    f_mu_24 = (prior24.prior_mean_flux - prior24.prior_flux_lower) / (prior24.prior_flux_upper - prior24.prior_flux_lower)
    f_sigma_24 = prior24.prior_sigma_flux / (prior24.prior_flux_upper - prior24.prior_flux_lower)

    f_mu_850 = (prior850.prior_mean_flux - prior850.prior_flux_lower) / (prior850.prior_flux_upper - prior850.prior_flux_lower)
    f_sigma_850 = prior850.prior_sigma_flux / (prior850.prior_flux_upper - prior850.prior_flux_lower)

    
    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior24.bkg[0], prior100.bkg[0],
                      prior160.bkg[0],prior250.bkg[0], prior350.bkg[0], prior500.bkg[0], prior850.bkg[0]],
        'bkg_prior_sig': [prior24.bkg[1], prior100.bkg[1],
                          prior160.bkg[1],prior250.bkg[1], prior350.bkg[1], prior500.bkg[1], prior850.bkg[1]],
        'conf_prior_sig': [0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'f_low_lim':[prior24.prior_flux_lower, prior100.prior_flux_lower, prior160.prior_flux_lower,
                     prior250.prior_flux_lower, prior350.prior_flux_lower, prior500.prior_flux_lower,
                     prior850.prior_flux_lower],
        'f_up_lim':[prior24.prior_flux_upper,prior100.prior_flux_upper,prior160.prior_flux_upper, 
                    prior250.prior_flux_upper,prior350.prior_flux_upper,prior500.prior_flux_upper, 
                   prior850.prior_flux_upper],
        'f_mu': [f_mu_24, f_mu_850], #f_mu
        'f_sigma': [f_sigma_24, f_sigma_850], #f_sigma
        'z_median': prior24.z_median,
        'z_sig': prior24.z_sig,
        # SPIRE
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        #MIPS
        'npix_mips24': prior24.snpix,
        'nnz_mips24': prior24.amat_data.size,
        'db_mips24': prior24.sim,
        'sigma_mips24': prior24.snim,
        'Val_mips24': prior24.amat_data,
        'Row_mips24': prior24.amat_row.astype(np.long),
        'Col_mips24': prior24.amat_col.astype(np.long),
        #PACS
        'npix_pacs100': prior100.snpix,
        'nnz_pacs100': prior100.amat_data.size,
        'db_pacs100': prior100.sim,
        'sigma_pacs100': prior100.snim,
        'Val_pacs100': prior100.amat_data,
        'Row_pacs100': prior100.amat_row.astype(np.long),
        'Col_pacs100': prior100.amat_col.astype(np.long),
        'npix_pacs160': prior160.snpix,
        'nnz_pacs160': prior160.amat_data.size,
        'db_pacs160': prior160.sim,
        'sigma_pacs160': prior160.snim,
        'Val_pacs160': prior160.amat_data,
        'Row_pacs160': prior160.amat_row.astype(np.long),
        'Col_pacs160': prior160.amat_col.astype(np.long),
        #LABOCA
        'npix_lb850': prior850.snpix,
        'nnz_lb850': prior850.amat_data.size,
        'db_lb850': prior850.sim,
        'sigma_lb850': prior850.snim,
        'Val_lb850': prior850.amat_data,
        'Row_lb850': prior850.amat_row.astype(np.long),
        'Col_lb850': prior850.amat_col.astype(np.long),

        
        #SEDs
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }


    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+IR_Submm_SED_GaussPrior'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)
    if control == True:
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    else:
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True, control = {"max_treedepth":12, "adapt_delta": 0.9})
        
    #return fit data
    return fit
