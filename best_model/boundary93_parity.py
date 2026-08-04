from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.interpolate import PchipInterpolator

class Boundary93ParityCorrection:
    """REQ6-identified diagnostic parity map for COMSOL PPR dp/dn.

    This is intentionally separate from the physical HK source.  The exported
    COMSOL PPR gradient is not numerically interchangeable with the derivative
    that best reproduces COMSOL pext on the present mesh, especially at 100 Hz.
    """
    def __init__(self, config: dict):
        self.config=config
        self.kind=str(config.get('kind',''))
        self.freq=np.asarray(config['anchor_frequencies_Hz'],float)
        if 'spline' in self.kind:
            self.theta=np.asarray(config['theta_rad'],float)
            self.delta=np.asarray([[complex(r,i) for r,i in zip(a['delta_real'],a['delta_imag'])] for a in config['anchors']])
        else:
            self.degree=int(config['degree'])
            self.alpha=np.asarray([complex(a['alpha_real'],a['alpha_imag']) for a in config['anchors']])
            self.coeff=np.asarray([[complex(r,i) for r,i in zip(a['coeff_real'],a['coeff_imag'])] for a in config['anchors']])
    @classmethod
    def from_json(cls,path:str|Path): return cls(json.loads(Path(path).read_text()))
    def _interp(self,f:float,values:np.ndarray):
        x=np.log(self.freq);xf=np.clip(math.log(float(f)),x[0],x[-1])
        if values.ndim==1:
            return np.interp(xf,x,values.real)+1j*np.interp(xf,x,values.imag)
        return np.array([np.interp(xf,x,values[:,j].real)+1j*np.interp(xf,x,values[:,j].imag) for j in range(values.shape[1])])
    def apply(self,freq_Hz:float,r:np.ndarray,z:np.ndarray,p:np.ndarray,dpdn:np.ndarray)->np.ndarray:
        theta=np.arctan2(z,r)
        ieq=np.argsort(np.abs(theta))[:min(5,len(theta))]
        p0=complex(np.mean(p[ieq]))
        if 'spline' in self.kind:
            delta_f=self._interp(freq_Hz,self.delta)
            dr=PchipInterpolator(self.theta,delta_f.real,extrapolate=True)(theta)
            di=PchipInterpolator(self.theta,delta_f.imag,extrapolate=True)(theta)
            return np.asarray(dpdn)+p0*(dr+1j*di)
        alpha=self._interp(freq_Hz,self.alpha);coeff=self._interp(freq_Hz,self.coeff)
        return alpha*np.asarray(dpdn)+p0*(C.chebvander(theta/(np.pi/2),self.degree)@coeff)
