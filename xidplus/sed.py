import pandas as pd
import numpy as np

class posterior_sed():
    def __init__(self, fit, priors,sed_prior_model,scale=True):
        import xidplus.stan_fit.stan_utility as stan_utility
        stan_utility.check_treedepth(fit)
        stan_utility.check_energy(fit)
        stan_utility.check_div(fit)
        self.nsrc = priors[0].nsrc
        self.samples = fit.extract()
        self.samples['src_f']=np.swapaxes(self.samples['src_f'], 1, 2)


        self.param_names = fit.model_pars
        self.summary=fit.summary()

        self.ID = priors[0].ID

        self.Rhat = {'src_f': fit.summary('src_f')['summary'][:, -1].reshape(priors[0].nsrc, len(priors)),
                     'sigma_conf': fit.summary('sigma_conf')['summary'][:, -1],
                     'bkg': fit.summary('bkg')['summary'][:, -1],'Nbb':fit.summary('Nbb')['summary'][:, -1],
                      'z':fit.summary('z')['summary'][:, -1]}

        self.n_eff = {'src_f': fit.summary('src_f')['summary'][:, -2].reshape(priors[0].nsrc, len(priors)),
                      'sigma_conf': fit.summary('sigma_conf')['summary'][:, -2],
                      'bkg': fit.summary('bkg')['summary'][:, -2],'Nbb':fit.summary('Nbb')['summary'][:, -2],
                      'z':fit.summary('z')['summary'][:, -2]}



class posterior_sed_fixz():
    def __init__(self, fit, priors,sed_prior_model,scale=True):
        import xidplus.stan_fit.stan_utility as stan_utility
        stan_utility.check_treedepth(fit)
        stan_utility.check_energy(fit)
        stan_utility.check_div(fit)
        self.nsrc = priors[0].nsrc
        self.samples = fit.extract()
        self.samples['src_f']=np.swapaxes(self.samples['src_f'], 1, 2)


        self.param_names = fit.model_pars
        self.summary=fit.summary()

        self.ID = priors[0].ID

        self.Rhat = {'src_f': fit.summary('src_f')['summary'][:, -1].reshape(priors[0].nsrc, len(priors)),
                     'sigma_conf': fit.summary('sigma_conf')['summary'][:, -1],
                     'bkg': fit.summary('bkg')['summary'][:, -1],'Nbb':fit.summary('Nbb')['summary'][:, -1]}

        self.n_eff = {'src_f': fit.summary('src_f')['summary'][:, -2].reshape(priors[0].nsrc, len(priors)),
                      'sigma_conf': fit.summary('sigma_conf')['summary'][:, -2],
                      'bkg': fit.summary('bkg')['summary'][:, -2],'Nbb':fit.summary('Nbb')['summary'][:, -2]}
        if scale is True:
            self.scale_posterior(priors)

#     def scale_posterior(self,priors):
#         #create indices for posterior (i.e. include backgrounds and sigma_conf)
#         """Stan searches over range 0-1 and scales parameters with flux limits. This function scales those parameters to flux values

#         :param priors: list of prior classes used in fit

#         """

#         for i in range(0,len(priors)):
#             lower=priors[i].prior_flux_lower
#             upper=priors[i].prior_flux_upper
#             self.samples['src_f'][:,i,:]=lower+(upper-lower)*self.samples['src_f'][:,i,:]


        

def berta_templates(SPIRE=True, PACS=True, MIPS=True, LABOCA=True):
    import os
    import numpy as np
    from astropy.io import ascii
    from scipy.interpolate import interp1d
    import xidplus

    temps = os.listdir(xidplus.__path__[0]+'/../test_files/templates_berta_norm_LIR/')

    #Generate Redshift Grid and convert to denominator for flux conversion(e.g. $4 \pi D_l ^ 2)$
    red = np.arange(0, 8, 0.01)
    red[0] = 0.000001
    from astropy.cosmology import Planck13
    import astropy.units as u
    div = (4.0 * np.pi * np.square(Planck13.luminosity_distance(red).cgs))
    div = div.value

    #Get appropriate filters
    from xidplus import filters
    filter = filters.FilterFile(file=xidplus.__path__[0] + '/../test_files/filters.res')
    
    SPIRE_250 = filter.filters[215]
    SPIRE_350 = filter.filters[216]
    SPIRE_500 = filter.filters[217]
    MIPS_24 = filter.filters[201]
    PACS_100 = filter.filters[250]
    PACS_160 = filter.filters[251]
    LABOCA_850 = filter.filters[307]

    bands = []  # wavelength [A]
    eff_lam=[]  # wavelength [um]
    if MIPS is True:
        bands.extend([MIPS_24])
        eff_lam.extend([24.0])

    if PACS is True:
        bands.extend([PACS_100,PACS_160])
        eff_lam.extend([100.0,160.0])

    if SPIRE is True:
        bands.extend([SPIRE_250,SPIRE_350,SPIRE_500])
        eff_lam.extend([250.0,350.0,500.0])

    if LABOCA is True:
        bands.extend([LABOCA_850])
        eff_lam.extend([850.0])

    print(eff_lam)
    import pandas as pd
    #read in berta's templates. Units:  wavelength: [0.1nm]  //  S(lambda), normalized to LIR=1L{sun}
    template = ascii.read(xidplus.__path__[0]+'/../test_files/templates_berta_norm_LIR/' + temps[0])
    #create dataframe with column==wavelength [um]
    df = pd.DataFrame(template['col1'].data / 1E4, columns=['wave'])
    SEDs = np.empty((len(temps), len(bands), red.size))
    for i in range(0, len(temps)):
        template = ascii.read(xidplus.__path__[0]+'/../test_files/templates_berta_norm_LIR/' + temps[i])
        # Add flux from templates to the dataframe (S(nu))
        df[temps[i]] = 1E26 * 3.826E33 * template['col2'] * (template['col1'] ** 2) / 3E18

        flux = template['col2'] * ((template['col1']) ** 2) / 3E18 # S(lambda)_norm * (lambda[Am]^2/c[Am/s]) --> S(nu)_norm
        wave = template['col1'] / 1E4  #lambda [um]
        
        for z in range(0, red.size):
            sed = interp1d((red[z] + 1.0) * wave, flux) # interpolate for different z. wave[um], S(nu)_norm
            for b in range(0, len(bands)):
                if eff_lam[b] == 850.0:
                    alpha = 0.0
                elif eff_lam[b] == 24.0:
                    alpha = -2.0
                else:
                    alpha = -1.0
                # for the function fnu_filt(sed_fnu,filt_nu,filt_trans,nu_0,sed_f0) 
                SEDs[i, b, z] = 1E26 * 3.826E33 * (1.0 + red[z]) * filters.fnu_filt(sed(bands[b].wavelength / 1E4),#[A to um]
                                                                                    # filt_nu: c[m/s] / lambda[(A to m)]
                                                                                    3E8 / (bands[b].wavelength / 1E10),
                                                                                    # filt_trans 
                                                                                    bands[b].transmission,
                                                                                    # nu_0: c[m/s] / lambda [(um to m)]
                                                                                    3E8 / (eff_lam[b] * 1E-6),alpha) / div[z]
    return SEDs, df




def swinbank_templates(SPIRE=True, PACS=True, MIPS=True, LABOCA=True):
    import os
    import numpy as np
    from astropy.io import ascii
    from scipy.interpolate import interp1d
    import xidplus
    from astropy.table import Table
    

    sw_templ = xidplus.__path__[0]+'/../test_files/swinbank_templates/all_templates.csv'
    temp_dust = Table.read(sw_templ)
    
    #Generate Redshift Grid and convert to denominator for flux conversion(e.g. $4 \pi D_l ^ 2)$
    red = np.arange(0, 8, 0.01)
    red[0] = 0.000001
    from astropy.cosmology import Planck13
    import astropy.units as u
    div = (4.0 * np.pi * np.square(Planck13.luminosity_distance(red).cgs))
    div = div.value

    #Get appropriate filters
    from xidplus import filters
    filter = filters.FilterFile(file=xidplus.__path__[0] + '/../test_files/filters.res')
    
    SPIRE_250 = filter.filters[215]
    SPIRE_350 = filter.filters[216]
    SPIRE_500 = filter.filters[217]
    MIPS_24 = filter.filters[201]
    PACS_100 = filter.filters[250]
    PACS_160 = filter.filters[251]
    LABOCA_850 = filter.filters[307]

    bands = []  # wavelength [A]
    eff_lam=[]  # wavelength [um]
    if MIPS is True:
        bands.extend([MIPS_24])
        eff_lam.extend([24.0])

    if PACS is True:
        bands.extend([PACS_100,PACS_160])
        eff_lam.extend([100.0,160.0])

    if SPIRE is True:
        bands.extend([SPIRE_250,SPIRE_350,SPIRE_500])
        eff_lam.extend([250.0,350.0,500.0])

    if LABOCA is True:
        bands.extend([LABOCA_850])
        eff_lam.extend([850.0])

    print(eff_lam)
               
    import pandas as pd
    #read in sw's templates. Units:  wavelength: [um]  //  S(nu)
    templates = temp_dust.columns[2:]
    #create dataframe with column==wavelength [um]
    df = pd.DataFrame(temp_dust['lambda'].data, columns=['wave'])
    SEDs = np.empty((len(templates), len(bands), red.size))
    c = 3E10
    for i in range(0, len(templates)):
        tmp_wave = temp_dust['lambda']
        tmp_name = templates[i].name
        tmp_flux = templates[i].data 
        # Calculate LIR
        LIR = np.trapz(tmp_flux[(tmp_wave>8) & (tmp_wave<1E3)],
                 x=tmp_wave[(tmp_wave>8) & (tmp_wave<1E3)])
        flux = tmp_flux/LIR
        df[tmp_name] = flux * 1E26 * 3.826E33 / c #
        #flux, normalized to LIR=1L{sun}
        flux = tmp_flux/LIR # 
        wave = tmp_wave #lambda [um]

        
        for z in range(0, red.size):
            sed = interp1d((red[z] + 1.0) * wave, flux) # interpolate for different z. wave[um], S(nu)_norm
            for b in range(0, len(bands)):
                if eff_lam[b] == 850.0:
                    alpha = 0.0
                elif eff_lam[b] == 24.0:
                    alpha = -2.0
                else:
                    alpha = -1.0
                # for the function fnu_filt(sed_fnu,filt_nu,filt_trans,nu_0,sed_f0) 
                SEDs[i, b, z] = 1E30 * 3.826E33 * (1.0 + red[z]) * filters.fnu_filt(sed(bands[b].wavelength / 1E4),#[A to um]
                                                                                    # filt_nu: c[m/s] / lambda[(A to m)]
                                                                                    3E8 / (bands[b].wavelength / 1E10),
                                                                                    # filt_trans 
                                                                                    bands[b].transmission,
                                                                                    # nu_0: c[m/s] / lambda [(um to m)]
                                                                                    3E8 / (eff_lam[b] * 1E-6),
                                                                                    # sed_f0
                                                                                    sed(eff_lam[b]),alpha) / div[z]
    return SEDs, df





## SEDs DaCunha
################

def dacunha_templates(SPIRE=True, PACS=True, MIPS=True, LABOCA=True):
    import os
    import numpy as np
    from astropy.io import ascii
    from scipy.interpolate import interp1d
    import xidplus
    from astropy.table import Table

    #Generate Redshift Grid and convert to denominator for flux conversion(e.g. $4 \pi D_l ^ 2)$
    red = np.arange(0, 8, 0.01)
    red[0] = 0.000001
    from astropy.cosmology import Planck13
    import astropy.units as u
    div = (4.0 * np.pi * np.square(Planck13.luminosity_distance(red).cgs))
    div = div.value

    #Get appropriate filters
    from xidplus import filters
    filter = filters.FilterFile(file=xidplus.__path__[0] + '/../test_files/filters.res')
    
    SPIRE_250 = filter.filters[215]
    SPIRE_350 = filter.filters[216]
    SPIRE_500 = filter.filters[217]
    MIPS_24 = filter.filters[201]
    PACS_100 = filter.filters[250]
    PACS_160 = filter.filters[251]
    LABOCA_850 = filter.filters[307]

    bands = []  # wavelength [A]
    eff_lam=[]  # wavelength [um]
    if MIPS is True:
        bands.extend([MIPS_24])
        eff_lam.extend([24.0])

    if PACS is True:
        bands.extend([PACS_100,PACS_160])
        eff_lam.extend([100.0,160.0])

    if SPIRE is True:
        bands.extend([SPIRE_250,SPIRE_350,SPIRE_500])
        eff_lam.extend([250.0,350.0,500.0])

    if LABOCA is True:
        bands.extend([LABOCA_850])
        eff_lam.extend([850.0])

    print(eff_lam)
    import pandas as pd
    #read in berta's templates. Units:  wavelength: [0.1nm]  //  S(lambda), normalized to LIR=1L{sun}
    template = Table.read( xidplus.__path__[0]+'/../test_files/daCunha_templates/templates_all.csv')
    #create dataframe with column==wavelength [um]
    df = pd.DataFrame(template[1][:])
    cols = template.columns[2:]
    SEDs = np.empty((len(cols), len(bands), red.size))
    c = 3E10
    
    for i in range(0, len(cols)):
        tmp_wave = template[1][:]
        tmp_name = cols[i].name
        tmp_flux = cols[i].data 
        # Calculate LIR
        LIR = np.trapz(tmp_flux[(tmp_wave>8) & (tmp_wave<1E3)],
                 x=tmp_wave[(tmp_wave>8) & (tmp_wave<1E3)])
        #flux, normalized to LIR=1L{sun}
        flux = tmp_flux/LIR
        df[tmp_name] = flux * 1E26 * 3.826E33 / c #
        wave = tmp_wave #lambda [um]


        for z in range(0,red.size):
            sed=interp1d((red[z]+1.0)*wave, flux)
            for b in range(0,len(bands)):
                if eff_lam[b]==850.0:
                    alpha = 0.0
                elif eff_lam[b]==24.0:
                    alpha = -2.0
                else:
                    alpha = -1.0
                SEDs[i,b,z]=1E26*3.826E33/c*(1.0+red[z])*filters.fnu_filt(sed(bands[b].wavelength/1E4), # [S(nu) template]
                                 3E10/(bands[b].wavelength/1E8),bands[b].transmission, # [bandpass , transmission filters]
                                 3E10/(eff_lam[b]*1E-4),alpha)/div[z] #[eff_lamda, alpha]
        
        
    return SEDs, df

