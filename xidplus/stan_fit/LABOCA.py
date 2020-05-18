from numpy import long

__author__ = 'mc741'
import os
import numpy as np
output_dir=os.getcwd()


full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)


stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'
def LABOCA_850(LABOCA_850,chains=4,iter=1000):
    """
    Fit the LABOCA 850 band

    :param LABOCA_850: xidplus.prior class
    :param chains: number of chains
    :param iter:  number of iterations
    :return: pystan fit object
    """

    #input data into a dictionary
    XID_data={'nsrc':LABOCA_850.nsrc,
              'f_low_lim':[LABOCA_850.prior_flux_lower],
              'f_up_lim':[LABOCA_850.prior_flux_upper],
              'bkg_prior':[LABOCA_850.bkg[0]],
              'bkg_prior_sig':[LABOCA_850.bkg[1]],
          'npix_lb850':LABOCA_850.snpix,
          'nnz_lb850':LABOCA_850.amat_data.size,
          'db_lb850':LABOCA_850.sim,
          'sigma_lb850':LABOCA_850.snim,
          'Val_lb850':LABOCA_850.amat_data,
          'Row_lb850': LABOCA_850.amat_row.astype(long),
          'Col_lb850': LABOCA_850.amat_col.astype(long)}

    #see if model has already been compiled. If not, compile and save it

    model_file= '/XID+LABOCA'
    from xidplus.stan_fit import get_stancode
    sm= get_stancode(model_file)
    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit
