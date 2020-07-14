import numpy as np
from astropy import wcs
from herschelhelp_internal.utils import inMoc
from xidplus import moc_routines


class prior_gauss(object):
    def cut_down_cat(self):
        """Cuts down prior class variables associated with the catalogue data to the MOC assigned to the prior class: self.moc
        """
        sgood = np.array(inMoc(self.sra, self.sdec, self.moc))

        self.sra = self.sra[sgood]
        self.sdec = self.sdec[sgood]
        self.nsrc = sum(sgood)
        self.ID = self.ID[sgood]
        if hasattr(self, 'nstack'):
            self.stack = self.stack[sgood]
            self.nstack = sum(self.stack)
        if hasattr(self, 'prior_flux_upper'):
            self.prior_flux_upper = self.prior_flux_upper[sgood]
        if hasattr(self, 'prior_flux_lower'):
            self.prior_flux_lower = self.prior_flux_lower[sgood]
            
    
    def __init__(self, ra, dec, flux_lower=None, flux_upper=None, ID=None, moc=None):
        
        """Input info for prior catalogue

        :param ra: Right ascension (JD2000) of sources
        :param dec: Declination (JD2000) of sources
        :param flux_lower: lower limit of flux for each source
        :param flux_upper: upper limit of flux for each source
        :param ID: HELP_ID for each source
        :param moc: Multi-Order Coverage map
        """
        # get positions of sources in terms of pixels
        if moc is None:
            cat_moc = moc_routines.create_MOC_from_cat(ra, dec)
        else:
            cat_moc = moc


        # Redefine prior list so it only contains sources in the map
        self.sra = ra
        self.sdec = dec
        self.nsrc = self.sra.size
        if flux_lower is None:
            flux_lower = np.full((ra.size), 0.00)
            flux_upper = np.full((ra.size), 1000.0)
        self.prior_flux_lower = flux_lower
        self.prior_flux_upper = flux_upper
        
        if ID is None:
            ID = np.arange(1, ra.size + 1, dtype='int64')
        self.ID = ID

        self.stack = np.full(self.nsrc, False)
        try:
            self.moc = self.moc.intersection(cat_moc)
        except AttributeError as e:
            self.moc=cat_moc

        self.cut_down_cat()
        
        
    def set_gaussprior_flux(self, f_mu, f_sigma):
        """Add gaussian parameters to define a prior on the flux

        :param f_mu: n array, where n is the number of sources, mean flux prior
        :param f_sigma: n array, sigma flux prior
        """

        self.prior_mean_flux = f_mu
        self.prior_sigma_flux = f_sigma
        