# distutils: language = c++
# distutils: sources = boost_invcdf.cpp
#cython: boundscheck=False
#cython: wraparound=False

cdef extern from "math.h":
    float exp(float theta)
    float sqrt(float theta)
    float fabs(float theta)
    float log(float theta)
    float erfc(float theta) 
    float floor(float theta)    
    float cos(float theta)  
    float sin(float theta)  
    float log1p(float theta)    

cdef extern from "limits.h":
    int RAND_MAX
    
from libc.math cimport M_PI
from libc.stdlib cimport rand

import sys
import scipy.stats as stats
import scipy.integrate
import scipy.linalg as la
import numpy as np
import time
cimport numpy as np
cimport cython

    
    
def logphi(double z, double sqrt2Pi, np.ndarray[double, ndim=1] c, np.ndarray[double, ndim=1] r, np.ndarray[double, ndim=1] q):

    cdef int i
    cdef double lp
    cdef double f = 0
    cdef double lp0 = -z/sqrt2Pi
    cdef double num = 0.5641895835477550741
    cdef double den = 1.0
    cdef double zDivSqrt2
    cdef double e

    #first case: close to zero
    if (z*z<0.0492):
        lp0 = -z/sqrt2Pi        
        for i in range(14): f = lp0*(c[i]+f)
        lp = -2*f - log(2)
        dlp = exp(-z**2/2.0 - lp) / sqrt2Pi
        
    #second case: very small
    elif (z<-11.3137):
        zDivSqrt2 = z/sqrt(2)
        for i in range(5): num = -zDivSqrt2*num + r[i] 
        for i in range(6): den = -zDivSqrt2*den + q[i]
        e = num/den
        lp = log(e/2.0) - z**2 / 2.0
        dlp = abs(den/num) * 2.0 / sqrt2Pi
        
    #third case: rest
    else:   
        lp = log(erfc(-z/sqrt(2)) / 2.0)
        dlp = exp(-z**2/2.0 - lp) / sqrt2Pi
        
    return lp, dlp
    
    




def likProbit_EP_single(double y, double mu, double s2, double sig2e, double t, np.ndarray[double, ndim=1] c, np.ndarray[double, ndim=1] rlogphi, np.ndarray[double, ndim=1] q, double sqrt2Pi):
    cdef double a,z,lZ, n_p, dlZ, d2lZ
    
    a = y / sqrt(s2+sig2e)
    z = a * (mu-t)  
    lZ, n_p = logphi(z, sqrt2Pi, c, rlogphi, q)    
    dlZ = a * n_p
    d2lZ = -a**2 * n_p * (z+n_p)
    return lZ, dlZ, d2lZ


    
    
def likFunc_EP_probit_asc_single(double mu, double s2, double logS0, double logSDiff, double sDiff, double sig2e, double t, np.ndarray[double, ndim=1] c, np.ndarray[double, ndim=1] rlogphi, np.ndarray[double, ndim=1] q, double sqrt2Pi):
    cdef double lZ, dlZ, d2lZ, logZstar, expDiff, temp, dZstar, d2Zstar

    lZ, dlZ, d2lZ = likProbit_EP_single(1, mu, s2, sig2e, t, c, rlogphi, q, sqrt2Pi)
    logZstar = np.logaddexp(logS0, logSDiff+lZ)
    expDiff = exp(lZ-logZstar)
    temp =  sDiff * expDiff
    dZstar  = temp * dlZ
    d2Zstar = temp * (d2lZ + dlZ**2 * (1-temp))
    return logZstar, dZstar, d2Zstar
    

    
    
def EP_innerloop_probit(np.ndarray[double, ndim=2] Sigma, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] mu,
                        np.ndarray[double, ndim=1] ttau, np.ndarray[double, ndim=1] tnu, double sig2e, np.ndarray[double, ndim=1] t):
                        
                        
    ########### init params to compute logphi #####################
    cdef np.ndarray[double, ndim=1] c = np.array([0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032,
    -0.0045563339802, 0.00556964649138, 0.00125993961762116,
    -0.01621575378835404, 0.02629651521057465, -0.001829764677455021,
    2*(1-M_PI/3.0), (4-M_PI)/3.0, 1, 1])
    
    cdef np.ndarray[double, ndim=1] rlogphi = np.array([1.2753666447299659525, 5.019049726784267463450,
    6.1602098531096305441, 7.409740605964741794425,
    2.9788656263939928886])
    
    cdef np.ndarray[double, ndim=1] q = np.array([ 2.260528520767326969592,  9.3960340162350541504,
    12.048951927855129036034, 17.081440747466004316 ,
    9.608965327192787870698,  3.3690752069827527677])
    
    cdef double sqrt2Pi = sqrt(2*M_PI)
    ###############################################################

    
    cdef np.ndarray[long, ndim=1] randpermN = np.random.permutation(range(y.shape[0]))
    cdef int i
    cdef double tau_ni, nu_ni, mu_ni, lZ, dlZ, d2lZ, ttau_old, tnu_old, dtt, dtn, ci
    cdef np.ndarray[double, ndim=1] sici, si
    
    for i in randpermN:     #iterate EP updates (in random order) over examples 
        tau_ni = 1.0/Sigma[i,i]  - ttau[i]              #Equation 3.56 rhs (and 3.66) from GP book
        nu_ni = (mu[i]/Sigma[i,i] - tnu[i])             #Equation 3.56 lhs (and 3.66) from GP book
        mu_ni = nu_ni / tau_ni      
        lZ, dlZ, d2lZ = likProbit_EP_single(y[i], mu_ni, 1.0/tau_ni, sig2e, t[i], c, rlogphi, q, sqrt2Pi)
        ttau_old, tnu_old = ttau[i], tnu[i]
        ttau[i] = -d2lZ  / (1+d2lZ/tau_ni)
        if (ttau[i] < 0): ttau[i]=0
        tnu[i]  = (dlZ - mu_ni*d2lZ ) / (1+d2lZ/tau_ni)
        dtt = ttau[i] - ttau_old
        dtn = tnu[i] - tnu_old
        si = Sigma[:,i]
        ci = dtt / (1+dtt*si[i])
        mu -= (ci* (mu[i]+si[i]*dtn) - dtn) * si                    #Equation 3.53 from GP book
        sici = si*ci
        Sigma -= sici[:, np.newaxis] * si[np.newaxis,:]

    return  ttau, tnu
    
    
    
    
    
def EP_innerloop_probit_both_parallel(np.ndarray[double, ndim=2] Sigma, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] mu,
                        double s0, double sDiff,
                        np.ndarray[double, ndim=1] ttau, np.ndarray[double, ndim=1] tnu, double sig2e, np.ndarray[double, ndim=1] t, update_freq, double step_size=1.0):
                        
                        
    ########### init params to compute logphi #####################
    cdef np.ndarray[double, ndim=1] c = np.array([0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032,
    -0.0045563339802, 0.00556964649138, 0.00125993961762116,
    -0.01621575378835404, 0.02629651521057465, -0.001829764677455021,
    2*(1-M_PI/3.0), (4-M_PI)/3.0, 1, 1])
    
    cdef np.ndarray[double, ndim=1] rlogphi = np.array([1.2753666447299659525, 5.019049726784267463450,
    6.1602098531096305441, 7.409740605964741794425,
    2.9788656263939928886])
    
    cdef np.ndarray[double, ndim=1] q = np.array([ 2.260528520767326969592,  9.3960340162350541504,
    12.048951927855129036034, 17.081440747466004316 ,
    9.608965327192787870698,  3.3690752069827527677])
    
    cdef double sqrt2Pi = sqrt(2*M_PI)
    ###############################################################

    cdef double logS0 = log(s0)
    cdef double logSDiff = log(sDiff)
    cdef np.ndarray[long, ndim=1] randpermN = np.random.permutation(range(y.shape[0]))
    cdef int i
    cdef double tau_ni, nu_ni, mu_ni, lZ, dlZ, d2lZ, ttau_old, tnu_old, dtt, dtn, ci
    cdef np.ndarray[double, ndim=1] sici, si
    
    cdef double lZ_numer, dlZ_numer, d2lZ_numer, lZ_denom, dlZ_denom, d2lZ_denom
    
    for i in randpermN:     #iterate EP updates (in random order) over examples 
        tau_ni = 1.0/Sigma[i,i]  - ttau[i]              #Equation 3.56 rhs (and 3.66) from GP book
        nu_ni = (mu[i]/Sigma[i,i] - tnu[i])             #Equation 3.56 lhs (and 3.66) from GP book
        mu_ni = nu_ni / tau_ni
        ttau_old, tnu_old = ttau[i], tnu[i]
        
        lZ_numer, dlZ_numer, d2lZ_numer = likProbit_EP_single(y[i], mu_ni, 1.0/tau_ni, sig2e, t[i], c, rlogphi, q, sqrt2Pi)     
        lZ_denom, dlZ_denom, d2lZ_denom = likFunc_EP_probit_asc_single(mu_ni, 1.0/tau_ni, logS0, logSDiff, sDiff, sig2e, t[i], c, rlogphi, q, sqrt2Pi)
        lZ = lZ_numer - lZ_denom
        dlZ = dlZ_numer - dlZ_denom
        d2lZ = d2lZ_numer - d2lZ_denom
        
        ttau[i] = step_size * (-d2lZ  / (1+d2lZ/tau_ni)) + (1-step_size) * ttau_old
        if (ttau[i] < 0):
            ttau[i]=0
            tnu[i]=0
        else: tnu[i]  = step_size * (dlZ - mu_ni*d2lZ ) / (1+d2lZ/tau_ni)  + (1-step_size) * tnu_old
        
        if (abs(ttau[i]) > 500): ttau[i]=500
        if (abs(tnu[i]) > 500): tnu[i]=(500 if tnu[i]>0 else -500)
        
        if (i>0 and i % update_freq == 0):
            dtt = ttau[i] - ttau_old
            dtn = tnu[i] - tnu_old
            si = Sigma[:,i]
            ci = dtt / (1+dtt*si[i])
            mu -= (ci* (mu[i]+si[i]*dtn) - dtn) * si                    #Equation 3.53 from GP book
            sici = si*ci
            Sigma -= sici[:, np.newaxis] * si[np.newaxis,:]

    return  ttau, tnu


   
    
