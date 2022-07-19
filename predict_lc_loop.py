import os
import torch
import sncosmo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import colors
from nn_twins_embedding import data, models, generate, CONFIG
from nn_twins_embedding.gp_math import nmad
from astropy.cosmology import FlatLambdaCDM, WMAP9

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

meta = pd.read_csv(os.path.join(CONFIG['data_dir'], 'meta.csv'),
                       index_col=0)
obs_wave = np.loadtxt(os.path.join(CONFIG['data_dir'], 'wave.dat'))
rf_fluxes = np.loadtxt(os.path.join(CONFIG['data_dir'], 'rf_fluxes.dat'))
rf_flux_errs = np.loadtxt(os.path.join(CONFIG['data_dir'], 'rf_flux_errs.dat'))

def get_sn_meta(sn_name):
    sn_meta = meta[meta.sn==sn_name]
    sn_meta = sn_meta[sn_meta.phase.between(-10, 40)]
    phases = sn_meta.phase.values
    spec_ids = sn_meta.index.values
    sn_meta = sn_meta[['z', 'dm', 'av', 'xi1', 'xi2', 'xi3']].mean()
    return sn_meta, phases, spec_ids

def plot_sn(sn_name, space=0.3):
    sn_meta, phases, spec_ids = get_sn_meta(sn_name)
    source = generate.TwinsEmbedding()
    model = sncosmo.Model(source=source)
    model.set(t0=0, **dict(sn_meta))
    model.set(z=0)
    wave = np.linspace(min(obs_wave), max(obs_wave), 200)
    plt.figure(figsize=(8, 10))
    for i, (spec_id, phase) in enumerate(zip(spec_ids, phases)):
        flux = model.flux(phase, wave)
        if i==0:
            plt.plot(wave, flux - i*space, color='C0', linewidth=2, label='Predicted')
            plt.plot(obs_wave, rf_fluxes[spec_id] - i*space, color='k',
                     linestyle='--', label='True')
        else:
            plt.plot(wave, flux - i*space, color='C0', linewidth=2)
            plt.plot(obs_wave, rf_fluxes[spec_id] - i*space, color='k',
                     linestyle='--')
        plt.text(8750, 0.05-i*space, '{:0.2f} days'.format(phase))
    plt.legend(fontsize=18, ncol=2, loc='upper right')
    plt.xscale('log')
    plt.xlim(3300, 10200)
    plt.title(sn_name)
    plt.xlabel('Rest-frame wavelength ($\AA$)')
    plt.ylabel('Flux + offset')
    plt.gca().set_xticks(np.arange(3500, 9500, 500))
    plt.gca().set_xticklabels(np.arange(3500, 9500, 500), rotation=45)
    plt.gca().set_yticks([])


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



def get_lightcurve1(sn):
    
    sn_meta, phases, spec_ids = get_sn_meta(sn)
    source = generate.TwinsEmbedding()
    model = sncosmo.Model(source=source)
    model.set(t0=0, av=sn_meta.av, xi1=sn_meta.xi1, xi2=sn_meta.xi2, xi3=sn_meta.xi3)
    model.set(dm=sn_meta.dm)
    model.set(z=0)
    model_phases = np.linspace(-10, 40, 100)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), gridspec_kw = {'height_ratios':[4,1]}, sharex='col')
    color_map = dict(zip('ubvri', 'mbgrk'))
    for band_name, band in generate.SNF_BANDS.items():
        model_lc = model.bandflux(band, model_phases)/1e15
        
        def chi_sq(model, phases, obs, err):
    
            model_phot = model.bandflux(band, phases)/1e15

            a = np.square(model_phot - obs)
            b = np.square(err)
            chi_sq = np.sum(a/b)/len(obs)

            return chi_sq
        
        ax1.plot(model_phases, model_lc,
                 color=color_map[band_name], label='SNf'+band_name+' (model)',
                 alpha=0.7)
        lc, lc_err = zip(*lc_from_spectra(band, rf_fluxes/1e15, phases, spec_ids, rf_flux_errs/1e15))
        ax1.errorbar(phases, lc, np.sqrt(lc_err), 
                      marker='o', elinewidth=1, linewidth=0,
                      c=color_map[band_name], label='SNf'+band_name+' (observed)')
        
        model_photopoint = model.bandflux(band, phases)/1e15
        residuals = model_photopoint - lc
        ax2.scatter(phases, residuals)
        meep = sn+' (chi_sq = '+str(chi_sq(model, phases, lc, np.sqrt(lc_err)))+')'
        
    ax1.legend(ncol=2)
    ax1.set_title(meep)
    ax2.set_xlabel('Phase (days)')
    ax1.set_ylabel('Flux')
    ax2.set_ylabel('residuals')
    yo = '/Users/FIRA FATMASIEFA/nn_twins_embedding/new plots4/'+str(sn)+'.jpg'
    plt.savefig(yo)
    plt.close()
    
    
    
    return True

def loop_over_sn(array):
    
    for i in tqdm(range(len(array))):
        get_lightcurve1(sne[i])
        
        

sne = np.unique(meta.sn)
loop_over_sn(sne)