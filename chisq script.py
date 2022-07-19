import os
import torch
import sncosmo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from nn_twins_embedding import data, models, generate, CONFIG
from nn_twins_embedding.gp_math import nmad
from astropy.cosmology import FlatLambdaCDM, WMAP9
from tqdm import tqdm

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

meta = pd.read_csv(os.path.join(CONFIG['data_dir'], 'meta.csv'),
                       index_col=0)
obs_wave = np.loadtxt(os.path.join(CONFIG['data_dir'], 'wave.dat'))
rf_fluxes = np.loadtxt(os.path.join(CONFIG['data_dir'], 'rf_fluxes.dat'))
rf_flux_errs = np.loadtxt(os.path.join(CONFIG['data_dir'], 'rf_flux_errs.dat'))


def lc_from_spectra(band, fluxes, phases, spec_ids, flux_errs=None):
    lc = []
    for i, (phase, spec_id) in enumerate(zip(phases, spec_ids)):
        flux = fluxes[spec_id]
        if flux_errs is not None:
            fluxerr = flux_errs[spec_id]
        else:
            fluxerr = None
        spec = sncosmo.Spectrum(wave=obs_wave, flux=flux, fluxerr=fluxerr, time=phase)
        lc.append(spec.bandfluxcov(band))
    return lc
    
    
    
sne = np.unique(meta.sn)

chisq_floor = []

for i in tqdm(range(len(sne))):

    sn_meta, phases, spec_ids = get_sn_meta(sne[i])
    source = generate.TwinsEmbedding()
    model = sncosmo.Model(source=source)
    model.set(t0=0, av=sn_meta.av, xi1=sn_meta.xi1, xi2=sn_meta.xi2, xi3=sn_meta.xi3)
    model.set(dm=sn_meta.dm)
    model.set(z=0)
    model_phases = np.linspace(-10, 40, 100)


    def chi_sq(model, phases):
        total = 0
        length = 0

        for band_name, band in generate.SNF_BANDS.items():

            lc, lc_err = zip(*lc_from_spectra(band, rf_fluxes/1e15, phases, spec_ids, rf_flux_errs/1e15))
            
            frac_err = np.sqrt(lc_err)/lc
            frac_err_floor = []
            
            for i in frac_err:
                if int(i) < 0.005:
                    frac_err_floor.append(0.005)

                else:
                    frac_err_floor.append(int(i))
                    
            obs_err_floor = []
            for i in range(0, len(frac_err_floor)):
                obs_err_floor.append(frac_err_floor[i]*lc[i])
            
            
            model_phot = model.bandflux(band, phases)/1e15

            a = np.square(model_phot - lc)
            b = np.square(obs_err_floor)

            chi_sq1 = np.sum(a/b)

            total += chi_sq1
            length += len(phases)

        chisq = total/length

        return chisq

    chisq_floor.append(chi_sq(model, phases))
    
chisq_floor