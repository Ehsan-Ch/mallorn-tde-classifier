"""Label-free descriptive flare fits and stacked multi-band shape summaries."""
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import skew,kurtosis
from .data import BANDS
from .features import dust_scale


def object_features(task):
    oid,z,ebv,time,flux,error,band=task
    z=max(float(z),0.)
    correction=np.array([dust_scale(ebv,b) for b in BANDS])
    flux=flux*correction[band];error=error*correction[band]
    core=np.isin(band,[1,2,3]);peak=time[np.flatnonzero(core)[np.argmax((flux/error)[core])]] if core.any() else time[np.argmax(flux/error)]
    t=(time-peak)/(1+z);out={};statistics={};normalized=np.zeros(len(flux));normerror=np.zeros(len(flux))
    for i,b in enumerate(BANDS):
        m=band==i;f=flux[m];e=error[m];tt=t[m]
        if not len(f):continue
        scale=max(np.max(f),np.median(e));normalized[m]=f/scale;normerror[m]=e/scale
        statistics[b]={'max':np.max(f),'q90':np.quantile(f,.9),'mean':np.mean(f),'std':np.std(f)}
        for k in [2,3,5,10,20]:
            use=f/e>k
            out[f'morph_{b}_snr{k}_count']=float(use.sum())
            out[f'morph_{b}_snr{k}_fraction']=float(use.mean())
            out[f'morph_{b}_snr{k}_span']=float(np.ptp(tt[use])) if use.sum()>1 else np.nan
        order=np.argsort(tt);tt,f,e=tt[order],f[order],e[order]
        for channel,values in [('flux',f),('snr',f/e)]:
            q=np.quantile(values,[.05,.1,.25,.5,.75,.9,.95])
            amp=max(q[-1]-q[0],1e-12)
            out[f'morph_{b}_{channel}_tail_asymmetry']=float((q[-1]+q[0]-2*q[3])/amp)
            out[f'morph_{b}_{channel}_iqr_ratio']=float((q[4]-q[2])/amp)
            out[f'morph_{b}_{channel}_kurtosis']=float(kurtosis(values)) if np.std(values)>1e-12 else np.nan
        for side,m in [('before',tt<0),('after',tt>=0),('late',tt>100)]:
            out[f'morph_{b}_{side}_mean_norm']=float(np.mean(f[m])/scale) if m.any() else np.nan
            out[f'morph_{b}_{side}_detected_fraction']=float(np.mean(f[m]/e[m]>3)) if m.any() else np.nan
    for a,b in combinations(BANDS,2):
        if a not in statistics or b not in statistics:continue
        for name in ['max','q90','mean','std']:
            fa,fb=statistics[a][name],statistics[b][name]
            out[f'morph_ratio_{a}_{b}_{name}']=float(np.log(fa/fb)) if min(fa,fb)>0 else np.nan
    # Morphological shape after independent per-band amplitude normalization.
    # Fit uses only this object's observations, never class labels.
    use=core&(t>-150)&(t<400)
    tt,ff,ee=t[use],normalized[use],normerror[use]
    out['morph_stacked_skew']=float(skew(ff)) if len(ff)>3 and np.std(ff)>1e-12 else np.nan
    out['morph_stacked_kurtosis']=float(kurtosis(ff)) if len(ff)>3 and np.std(ff)>1e-12 else np.nan
    if len(tt)>=10:
        # Floor tiny uncertainties to avoid allowing a single point to dominate
        # a descriptive fit with deliberately simplified spectral evolution.
        ee=np.maximum(ee,.03)
        for kind in ['exponential','powerlaw']:
            def predict(params):
                center,rise,fall,amplitude,baseline=params
                phase=tt-center
                before=np.exp(-.5*(np.minimum(phase,0)/rise)**2)
                after=np.exp(-np.maximum(phase,0)/fall) if kind=='exponential' else (1+np.maximum(phase,0)/fall)**(-5/3)
                return baseline+amplitude*before*after
            fit=least_squares(lambda p:(predict(p)-ff)/ee,[0,20,50,1,0],
                              bounds=([-60,1,1,0,-.5],[60,300,1000,3,1]),max_nfev=100,loss='soft_l1')
            for name,value in zip(['center','rise','fall','amplitude','baseline'],fit.x):out[f'flare_{kind}_{name}']=float(value)
            out[f'flare_{kind}_chi2']=float(np.mean(((predict(fit.x)-ff)/ee)**2))
            out[f'flare_{kind}_asymmetry']=float(fit.x[2]/fit.x[1])
            out[f'flare_{kind}_success']=float(fit.success)
        out['flare_powerlaw_exp_chi2_ratio']=out['flare_powerlaw_chi2']/max(out['flare_exponential_chi2'],1e-12)
    return oid,out
