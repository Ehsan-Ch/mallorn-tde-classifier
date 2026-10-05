"""Independent per-object GP and thermal-shape features for local experiments.

Inspired by Avocado and published MALLORN approaches; no external feature
tables/code are used. Monochromatic thermal fits are descriptive approximations.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis

from .data import BANDS
from .features import WAVELENGTHS, dust_scale

PHASES = (-20, -10, 0, 10, 15, 30, 60, 90)


def luminosity_distance_cm(z):
    """Flat LCDM, H0=70 km/s/Mpc, Omega_m=.3, radiation neglected."""
    if z <= 0:
        return np.nan
    integral = quad(lambda zz: 1 / np.sqrt(.3*(1+zz)**3+.7), 0, z)[0]
    return (1+z)*299792.458/70*integral*3.085677581491367e24


def kernel(t1, w1, t2, w2, tau):
    # Matern 3/2 over time and log wavelength; fixed wavelength scale .4.
    distance = np.sqrt(((t1[:,None]-t2[None,:])/tau)**2 +
                       ((w1[:,None]-w2[None,:])/.4)**2)
    r = np.sqrt(3)*distance
    return (1+r)*np.exp(-r)


def thermal_fit(flux, error, z, distance):
    valid = np.isfinite(flux)&np.isfinite(error)&(flux>2*error)&(error>0)
    if valid.sum()<3 or not np.isfinite(distance):
        return np.full(4,np.nan)
    # f_nu=(1+z)*pi*B_nu((1+z)nu_obs,T)*(R/D_L)^2.
    nu = 2.99792458e18/np.array([WAVELENGTHS[b] for b in BANDS])[valid]*(1+z)
    f, e = flux[valid]*1e-29, error[valid]*1e-29  # microJy -> cgs
    temperatures = np.geomspace(3500,100000,96)
    bb = 2*6.62607015e-27*nu[None,:]**3/(2.99792458e10**2)/np.expm1(
        6.62607015e-27*nu[None,:]/(1.380649e-16*temperatures[:,None]))
    shape = np.pi*bb*(1+z)
    amplitude = np.sum(shape*f/e**2,axis=1)/np.sum(shape**2/e**2,axis=1)
    chi = np.sum(((f-shape*amplitude[:,None])/e)**2,axis=1)
    i = int(np.argmin(chi))
    radius = distance*np.sqrt(amplitude[i])
    luminosity = 4*np.pi*radius**2*5.670374419e-5*temperatures[i]**4
    return np.array([np.log10(temperatures[i]),np.log10(radius),np.log10(luminosity),chi[i]/max(1,valid.sum()-2)])


def object_features(task):
    object_id,z,ebv,time,flux,error,band = task
    z=max(float(z),0.)
    factors=np.array([dust_scale(ebv,b) for b in BANDS])
    flux=flux*factors[band];error=error*factors[band]
    snr=flux/error
    reference=np.flatnonzero(np.isin(band,[1,2,3]))
    if not len(reference):reference=np.arange(len(time))
    peak=time[reference[np.argmax(snr[reference])]]
    t=(time-peak)/(1+z)
    wavelengths=np.log(np.array([WAVELENGTHS[b] for b in BANDS]))
    distance=luminosity_distance_cm(z)
    out={'phys_log_distance':float(np.log10(distance)) if distance>0 else np.nan}
    for bi,b in enumerate(BANDS):
        mask=band==bi; f=flux[mask];tt=t[mask];ee=error[mask]
        if not len(f):continue
        order=np.argsort(tt);f=f[order];tt=tt[order];ee=ee[order]
        peak_flux=max(float(np.max(f)),1e-10)
        for name,val in [('mean',np.mean(f)),('q90',np.quantile(f,.9)),('peak',np.max(f))]:
            out[f'phys_{b}_logL_{name}']=float(np.log10(4*np.pi*distance**2*val*1e-29/(1+z))) if val>0 and distance>0 else np.nan
        out[f'phys_{b}_qmax_q50']=float(np.max(f)/max(abs(np.median(f)),np.median(ee)))
        out[f'phys_{b}_q90_q50']=float(np.quantile(f,.9)/max(abs(np.median(f)),np.median(ee)))
        out[f'phys_{b}_excess_variance']=float((np.var(f)-np.mean(ee**2))/peak_flux**2)
        out[f'phys_{b}_von_neumann']=float(np.mean(np.diff(f)**2)/max(np.var(f),1e-15)) if len(f)>2 else np.nan
        # Equal-time resampling within observing seasons, never across >60-day gaps.
        values=[]
        for segment in np.split(np.arange(len(tt)),np.flatnonzero(np.diff(tt)>60/(1+z))+1):
            if len(segment)<2:continue
            grid=np.linspace(tt[segment[0]],tt[segment[-1]],max(3,min(100,int(np.ptp(tt[segment])/3)+1)))
            values.extend(np.interp(grid,tt[segment],f[segment]))
        if len(values)>3 and np.std(values)>1e-12:
            out[f'phys_{b}_resampled_skew']=float(skew(values))
            out[f'phys_{b}_resampled_kurtosis']=float(kurtosis(values))
        for left,right in [(-80,-30),(-30,-10),(-10,10),(10,30),(30,60),(60,120),(120,300)]:
            m=(tt>=left)&(tt<right)
            out[f'phys_{b}_bin{left}_{right}_mean']=float(np.mean(f[m])) if m.any() else np.nan
            out[f'phys_{b}_bin{left}_{right}_norm']=float(np.mean(f[m])/peak_flux) if m.any() else np.nan
    # Cap computation deterministically, retaining time coverage and strong detections.
    order=np.argsort(t,kind='stable')
    if len(order)>160:
        strong=np.argsort(snr)[-40:]
        uniform=order[np.linspace(0,len(order)-1,120).astype(int)]
        order=np.unique(np.r_[strong,uniform])
    gt,gw=t[order],wavelengths[band[order]]
    scale=max(float(np.quantile(np.abs(flux),.95)),float(np.median(error)),1e-8)
    gy=flux[order]/scale
    variance=(error[order]/scale)**2+1e-6
    def fit(log_tau,return_fit=False):
        tau=np.exp(log_tau)
        cov=kernel(gt,gw,gt,gw,tau);cov.flat[::len(cov)+1]+=variance
        try:
            chol=cho_factor(cov,lower=True,check_finite=False)
            alpha=cho_solve(chol,gy,check_finite=False)
            loss=.5*float(gy@alpha)+float(np.log(np.diag(chol[0])).sum())
        except np.linalg.LinAlgError:
            return np.inf
        return (tau,chol,alpha,loss) if return_fit else loss
    opt=minimize_scalar(fit,bounds=(np.log(2),np.log(500)),method='bounded',options={'xatol':.1,'maxiter':20})
    tau,chol,alpha,loss=fit(opt.x,True)
    out.update(gp_log_timescale=float(np.log10(tau)),gp_nll_per_obs=float(loss/len(gt)))
    def predict(qt,qb,uncertainty=False):
        cov=kernel(gt,gw,qt,wavelengths[qb],tau)
        mean=cov.T@alpha*scale
        if not uncertainty:return mean
        var=np.maximum(1-np.sum(cov*cho_solve(chol,cov,check_finite=False),axis=0),1e-8)
        return mean,np.sqrt(var)*scale
    grid=np.linspace(float(t.min()),float(t.max()),min(256,max(40,int(np.ptp(t)/3)+1)))
    qtime=np.tile(grid,6);qband=np.repeat(np.arange(6),len(grid))
    smooth=predict(qtime,qband).reshape(6,-1)
    # Peak selected using supported g/r/i grid locations, avoiding unconstrained gaps.
    support=np.min(abs(grid[:,None]-t[None,:]),axis=1)<30/(1+z)
    combined=np.maximum(smooth[1:4],0).sum(axis=0)
    combined[~support]=-np.inf
    peak2=float(grid[int(np.argmax(combined))])
    out['gp_peak_shift']=peak2
    phase_grid=grid-peak2
    for bi,b in enumerate(BANDS):
        s=smooth[bi];maximum=float(np.max(s));normal=max(maximum,1e-10)
        out[f'gp_{b}_peak']=maximum
        out[f'gp_{b}_peak_phase']=float(phase_grid[np.argmax(s)])
        out[f'gp_{b}_mean_norm']=float(np.mean(s)/normal)
        out[f'gp_{b}_std_norm']=float(np.std(s)/normal)
        out[f'gp_{b}_skew']=float(skew(s)) if np.std(s)>1e-12 else np.nan
        out[f'gp_{b}_logLpeak']=float(np.log10(4*np.pi*distance**2*maximum*1e-29/(1+z))) if maximum>0 and distance>0 else np.nan
        for frac in [.2,.5,.8]:
            above=phase_grid[s>frac*normal]
            out[f'gp_{b}_width{frac}']=float(np.ptp(above)) if len(above)>1 else np.nan
        # Is most of the integrated positive flux concentrated near a single peak?
        near=abs(phase_grid)<60
        out[f'gp_{b}_concentration']=float(np.maximum(s[near],0).sum()/max(np.maximum(s,0).sum(),1e-10))
    qt=np.tile(np.array(PHASES)+peak2,6);qb=np.repeat(np.arange(6),len(PHASES))
    means,errors=predict(qt,qb,True);means=means.reshape(6,-1);errors=errors.reshape(6,-1)
    # Inflate very small conditional uncertainty by a representative measurement error.
    errors=np.sqrt(errors**2+np.median(error)**2)
    for bi,b in enumerate(BANDS):
        for pi,phase in enumerate(PHASES):
            out[f'gp_{b}_p{phase}_norm']=float(means[bi,pi]/max(np.max(smooth[bi]),1e-10))
            out[f'gp_{b}_p{phase}_snr']=float(means[bi,pi]/errors[bi,pi])
    for a,b in combinations(range(6),2):
        colors=[]
        for pi,phase in enumerate(PHASES):
            reliable=(means[[a,b],pi]>2*errors[[a,b],pi]).all()
            color=float(-2.5*np.log10(means[a,pi]/means[b,pi])) if reliable else np.nan
            out[f'gp_color_{BANDS[a]}{BANDS[b]}_p{phase}']=color;colors.append(color)
        out[f'gp_color_{BANDS[a]}{BANDS[b]}_evolution']=colors[5]-colors[2]
    thermal=[]
    for pi,phase in enumerate(PHASES):
        pars=thermal_fit(means[:,pi],errors[:,pi],z,distance);thermal.append(pars)
        for name,value in zip(['logT','logR','logL','chi2'],pars):out[f'bb_{name}_p{phase}']=float(value)
    for idx,name in enumerate(['logT','logR','logL']):
        out[f'bb_{name}_evolution10_30']=float(thermal[5][idx]-thermal[3][idx])
        out[f'bb_{name}_evolution0_60']=float(thermal[6][idx]-thermal[2][idx])
    return object_id,out
