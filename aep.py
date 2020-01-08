import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import scipy.optimize as optimize
import scipy.integrate as integrate
import sklearn.linear_model
import kernels
import ep_fast
#import EP_cython
np.set_printoptions(precision=4, linewidth=200)


class GradientFields():
    def __init__(self, K_nodiag, s0, t_i, prev):
        normPDF = stats.norm(0,1)
        
        try: t_i[0]
        except: t_i = np.zeros(K_nodiag.shape[0]) + t_i

        #general computations (always the same if the fixed effects are 0!!!!!)     
        self.Ki = normPDF.sf(t_i)       
        self.Ps = s0 + (1-s0)*self.Ki
        self.Pi = self.Ki / self.Ps     
        self.stdY = np.sqrt(self.Pi * (1-self.Pi))

        #compute Atag0 and B0
        self.phi_ti = normPDF.pdf(t_i)      
        self.phitphit = np.outer(self.phi_ti, self.phi_ti)
        self.stdY_mat = np.outer(self.stdY, self.stdY)
        mat1_temp = self.phi_ti / self.stdY
        self.mat1 = np.outer(mat1_temp, mat1_temp)
        sumProbs_temp = np.tile(self.Pi, (K_nodiag.shape[0], 1))
        sumProbs = sumProbs_temp + sumProbs_temp.T
        Atag0_B0_inner_vec = self.Pi*(1-s0)
        self.mat2 = np.outer(Atag0_B0_inner_vec, Atag0_B0_inner_vec) + 1-sumProbs*(1-s0)
        self.Atag0 = self.mat1*self.mat2
        self.B0 = np.outer(self.Ps, self.Ps)
        
        #Compute the elements of the function value (the squared distance between the observed and expected pairwise phenotypic covariance)
        self.K_nodiag_AB0 = K_nodiag * self.Atag0/self.B0
        self.K_nodiag_sqr_AB0 = K_nodiag * self.K_nodiag_AB0

class PrevTest():
    def __init__(self, n, m, prev, useFixed, h2Scale=1.0, prng=None, num_generate=None):
    
        self.prng = prng
        if (prng is None): self.prng = np.random.RandomState(args.seed)
    
        self.n = n
        self.useFixed = useFixed
        self.h2Scale = h2Scale
        
        if num_generate is None:
            if prev == 0.5:
                numGeno = n
            else:
                numGeno = np.maximum(int(float(self.n)/float(2*prev)), 25000)
        else: 
            numGeno = num_generate
        
        #generate SNPs
        mafs = self.prng.rand(m) * 0.45 + 0.05
        self.X  = prng.binomial(2, mafs, size=(numGeno, m)).astype(np.float)           
        mafs_estimated = mafs.copy()
                            
        self.X_estimated = self.X.copy()            
        
        self.X -= 2*mafs
        self.X_estimated -= 2*mafs_estimated
        
        self.X /= np.sqrt(2*mafs*(1-mafs))
        self.X_estimated /= np.sqrt(2*mafs_estimated*(1-mafs_estimated))
        self.m = m
        self.n = n
        
        X_mean_diag = np.mean(np.einsum('ij,ij->i', self.X, self.X)) / self.X.shape[1]
        X_estimated_mean_diag = np.mean(np.einsum('ij,ij->i', self.X_estimated, self.X_estimated)) / self.X.shape[1]
        self.diag_ratio = X_estimated_mean_diag / X_mean_diag
        
        self.prev = prev
        
        #approx coeffs lam_i and c_i for logistic likelihood
        self.logistic_c = np.array([1.146480988574439e+02, -1.508871030070582e+03, 2.676085036831241e+03, -1.356294962039222e+03,  7.543285642111850e+01])
        self.logistic_lam = np.sqrt(2)*np.array([0.44 ,0.41, 0.40, 0.39, 0.36])
        self.logistic_lam2 = self.logistic_lam**2
        self.logistic_clam = self.logistic_c * self.logistic_lam
        

    def genData(self, h2, eDist, numFixed, ascertain=True, scaleG=False, extraSNPs=0, fixedVar=0, frac_cases=0.5, kernel='linear', rbf_scale=1.0):
    
    
        args.seed += 1    
        self.true_h2 = h2
    
        self.ascertain = ascertain
        self.eDist = eDist
        
        if (numFixed==0): fixedVar=0
        if (numFixed > 0): assert fixedVar>0
        self.fixedVar = fixedVar
        self.covars = self.prng.randn(self.X.shape[0], numFixed)
            
        if (eDist == 'normal' and not scaleG): sig2g = h2/(1-h2)
        elif (eDist == 'normal' and scaleG): sig2g = h2
        elif (eDist == 'logistic' and not scaleG): sig2g = (np.pi**2)/3.0 * h2 / (1 - h2)
        elif (eDist == 'logistic' and scaleG): sig2g = h2
        else: raise ValueError('unknown e_dist. Valid value are normal, logistic')
    
        if kernel == 'linear':
            self.beta = self.prng.randn(self.m) * np.sqrt(sig2g/self.m)         #generate effect sizes
            self.g = self.X.dot(self.beta)                                      #generate genetic effects
            self.g_estimated = self.X_estimated.dot(self.beta)
        elif args.kernel == 'rbf':
            assert scaleG
            kernel_obj = kernels.ScaledKernel(kernels.RBFKernel(self.X))
            K = kernel_obj.getTrainKernel(np.array([np.log(rbf_scale), np.log(sig2g) / 2.0]))
            L = la.cholesky(K, lower=True, overwrite_a=True)                
            self.g = L.dot(np.random.randn(K.shape[0]))                
            
            if np.allclose(self.X, self.X_estimated):
                self.g_estimated = self.g.copy()
            else:
                kernel_obj_estimated = kernels.ScaledKernel(kernels.RBFKernel(self.X_estimated))
                K_estimated = kernel_obj_estimated.getTrainKernel(np.array([np.log(rbf_scale), np.log(sig2g) / 2.0]))
                L_estimated = la.cholesky(K_estimated, lower=True, overwrite_a=True)
                self.g_estimated = L_estimated.dot(np.random.randn(K_estimated.shape[0]))
        else:
            raise ValueError('unknown kernel')
            
        #create identical twins if needed
        if self.prev == 0.5:
            numGeno = self.n
        else:
            numGeno = np.maximum(int(float(self.n)/float(2*self.prev)), 25000)
            
        self.fixedEffects = np.ones(numFixed) * (0 if (numFixed==0) else np.sqrt(fixedVar / numFixed))
        self.covars = self.prng.randn(self.g.shape[0], numFixed)
        
        m = self.covars.dot(self.fixedEffects)          
        self.g += m
        self.g_estimated += m
        
        if (eDist == 'logistic' and numFixed>0): raise ValueError('logistic distribution with fixed effects not supported')
    
        #generate environmental effect      
        if (eDist == 'normal' and not scaleG): e = self.prng.randn(self.g.shape[0])
        elif (eDist == 'normal' and scaleG): e = self.prng.randn(self.g.shape[0]) * np.sqrt(1 - sig2g)# - (fixedVar if (numFixed>0) else 0))
        elif (eDist == 'logistic' and not scaleG): e = stats.logistic(0,1).rvs(self.g.shape[0])
        elif (eDist == 'logistic' and scaleG): e = stats.logistic(0,1).rvs(self.g.shape[0]) * np.sqrt(1-sig2g) / np.sqrt((np.pi**2)/3.0)
        else: raise ValueError('unknown e distribution: ' + self.eDist)
        self.yAll = self.g + e
        self.yAll_estimated = self.g_estimated + e
        
        self.affCutoff = np.percentile(self.yAll, 100*(1-self.prev))
        cases = (self.yAll >= self.affCutoff)                               #determine cases
        cases_estimated = (self.yAll_estimated >= self.affCutoff)                               #determine cases
        controls = ~cases
        controls_estimated = ~cases_estimated
        self.y = np.ones(self.yAll.shape[0])
        self.y[controls] = -1
        self.y_estimated = np.ones(self.yAll.shape[0])
        self.y_estimated = np.ones(self.yAll.shape[0])
        self.y_estimated[controls_estimated] = -1
        
        #select cases and controls      
        caseInds = np.where(cases)[0]
        controlInds = np.where(controls)[0]
        if ascertain:
            numCases = np.sum(cases)
            if (numCases > self.n/2+2):
                selectedCases = self.prng.permutation(numCases)[:self.n//2]
                caseInds = caseInds[selectedCases]
                numCases = len(caseInds)
            numControls = int(numCases * (1-frac_cases)/frac_cases)
            selectedControls = self.prng.permutation(controls.sum())[:numControls]
            selectedInds = np.concatenate((caseInds, controlInds[selectedControls]))
        else:
            while True:
                selectedInds = self.prng.permutation(cases.shape[0])[:self.n]
                if (np.sum(cases[selectedInds]) > 0): break

        #scramble inds to avoid numerical issues        
        self.prng.shuffle(selectedInds)
        
        self.y = self.y[selectedInds]       
        ###print('%%cases: %0.2f'%(np.mean(self.y>0)))
        self.g = self.g[selectedInds]
        self.g_estimated = self.g_estimated[selectedInds]
        self.y_cont = self.yAll[selectedInds]
        self.covars = self.covars[selectedInds, :]
        self.X_selected = self.X_estimated[selectedInds, :]

        
        if (extraSNPs > 0):
            ###print('Adding', extraSNPs, 'non-causal SNPs...')
            mafs = self.prng.rand(extraSNPs) * 0.45 + 0.05
            X2  = self.prng.binomial(2, mafs, size=(self.X_selected.shape[0], extraSNPs)).astype(np.float)
            X2 -= 2*mafs
            X2 /= np.sqrt(2*mafs*(1-mafs))          
            self.X_selected = np.concatenate((self.X_selected, X2), axis=1)
                
        

        #create the kernel matrix
        if kernel=='linear':
            kernel_obj = kernels.linearKernel(self.X_selected)                
            K = kernel_obj.getTrainKernel(np.array([]))
        elif kernel=='rbf':
            kernel_obj = kernels.RBFKernel(self.X_selected)
            K = kernel_obj.getTrainKernel(np.array([np.log(rbf_scale)]))
        else:
            raise ValueError('unknown kernel')
            
        self.kernel = kernels.ScaledKernel(kernel_obj)

        


    def computeT(self, K, sig2e=np.pi**2/3.0):
    
        if (self.prev==0.5): return 0.0
    
        controls = (self.y < 0)
        cases = ~controls       
        diagK = np.diag(K)
        sig2g = (1-self.prev)*np.mean(diagK[controls]) + self.prev*np.mean(diagK[cases])
    
        if (self.eDist == 'normal'): t = stats.norm(0, np.sqrt(sig2g+1)).isf(self.prev)
        elif (self.eDist == 'logistic'):
            s = np.sqrt(3*sig2e/np.pi**2)
            normCache = np.log(np.sqrt(2*np.pi*sig2g))
            llF = lambda f,t: -(f-t)**2/(2*sig2g) - normCache
            pFAndY = lambda f,t: np.exp(llF(f,t)) * (1.0/(1+np.exp(-f/s)) if f>-35 else 0.0)
            pY = lambda t: integrate.quad(lambda f:pFAndY(f,t), -np.inf, np.inf)
            t = -optimize.minimize_scalar(lambda t:(pY(t)[0]-self.prev)**2, method='bounded', bounds=(-8, 8)).x
        else: raise Exception('unknown e distribution: ' + self.eDist)
            
        return t
        
        
    def likErf_EP(self, y, mu, s2, hyp=None, compDerivs=False):
    
        sqrtVarDenom = 1.0 / np.sqrt(1+s2)          
        z = mu * sqrtVarDenom * y       
        normPDF = stats.norm(0,1)
        lZ = normPDF.logcdf(z)
        if (not compDerivs): return lZ
        
        n_p = np.exp(normPDF.logpdf(z) - lZ)
        dlZ = n_p * sqrtVarDenom * y                    #1st derivative wrt mean        
        d2lZ = -n_p * (z+n_p) / (1+s2)                  #2nd derivative wrt mean
        
        return lZ, dlZ, d2lZ
        
        
        
    #compute EP for a single individual, and compute derivatives with respect to the mean (mu)
    def likLogistic_EP_single_new(self, y, mu, s2, hyp):
    
        t = hyp[4]; mu = mu-t
        hyp[4] = 0      
        lZc, dlZc, d2lZc = self.likProbit_EP_single(y, mu*self.logistic_lam, s2*self.logistic_lam2, hyp)
        lZ = self.log_expA_x_single(lZc, self.logistic_c)                               #A=lZc, B=dlZc, d=c.*lam', lZ=log(exp(A)*c)
        dlZ  = self.expABz_expAx_single(lZc, self.logistic_c, dlZc, self.logistic_clam)     #((exp(A).*B)*d)./(exp(A)*c)        
        #d2lZ = ((exp(A).*Z)*e)./(exp(A)*c) - dlZ.^2 where e = c.*(lam.^2)'
        d2lZ = self.expABz_expAx_single(lZc, self.logistic_c, dlZc**2+d2lZc, self.logistic_c * self.logistic_lam2) - dlZ**2

        
        #A note (from the GPML package documentation):
        #The scale mixture approximation does not capture the correct asymptotic
        #behavior; we have linear decay instead of quadratic decay as suggested
        #by the scale mixture approximation. By observing that for large values 
        #of -f*y ln(p(y|f)) for likLogistic is linear in f with slope y, we are
        #able to analytically integrate the tail region.
        val = np.abs(mu) - 196/200*s2-4                         #empirically determined bound at val==0
        lam = 1.0 / (1.0+np.exp(-10*val))                       #interpolation weights
        lZtail = np.minimum(s2/2.0-np.abs(mu), -0.1)            #apply the same to p(y|f) = 1 - p(-y|f)     
        if (mu*y > 0):
            lZtail = np.log(1-np.exp(lZtail))       #label and mean agree           
            dlZtail = 0
        else:
            dlZtail = -np.sign(mu)
        lZ   = (1-lam)*  lZ + lam*  lZtail                      #interpolate between scale ..
        dlZ  = (1-lam)* dlZ + lam* dlZtail                      #..  mixture and   ..
        d2lZ = (1-lam)*d2lZ                                     #.. tail approximation

        hyp[4] = t
        return lZ, dlZ, d2lZ
        
        
    def likLogistic_EP_multi_new(self, y, mu, s2, hyp=None):

        t = hyp[4]; mu = mu-t
        hyp[4] = 0
        
        lZc = self.likProbit_EP_multi(np.outer(y, np.ones(5)), np.outer(mu, self.logistic_lam), np.outer(s2, self.logistic_lam2), hyp)
        lZ = self.log_expA_x_multi(lZc, self.logistic_c)                            #A=lZc, B=dlZc, d=c.*lam', lZ=log(exp(A)*c)
        
        val = np.abs(mu) - 196/200*s2-4                         #empirically determined bound at val==0
        lam = 1.0 / (1.0+np.exp(-10*val))                       #interpolation weights
        lZtail = np.minimum(s2/2.0-np.abs(mu), -0.1)            #apply the same to p(y|f) = 1 - p(-y|f)
        muy = mu*y
        id = muy>0; lZtail[id] = np.log(1-np.exp(lZtail[id]))   #label and mean agree
        lZ   = (1-lam)*lZ + lam*lZtail                          #interpolate between scale mixture and tail approximation
        
        hyp[4] = t
        
        return lZ
        
        
        
        
        
    def likProbit_EP_multi(self, y, mu, s2, hyp):
        sig2e, t = hyp[0], hyp[4]
        lZ = stats.norm(0,1).logcdf(y * (mu-t) / np.sqrt(s2+sig2e))
        return lZ
        
    def likProbit_EP_single(self, y, mu, s2, hyp):
        sig2e, t = hyp[0], hyp[4]       
        a = y / np.sqrt(s2+sig2e)       
        z = a * (mu-t)
        normPDF = stats.norm(0,1)
        lZ = normPDF.logcdf(z)      
        n_p = np.exp(normPDF.logpdf(z) - lZ)
        dlZ = a * n_p
        d2lZ = -a**2 * n_p * (z+n_p)
        return lZ, dlZ, d2lZ

    def likFunc_EP_asc_multi(self, y, mu, s2, hyp):
        logS0, logSDiff, sDiff = hyp[1], hyp[2], hyp[3]
        likFunc_numer, likFunc_denom = hyp[5], hyp[6]
        lZ = likFunc_numer(1, mu, s2, hyp)
        logZstar = np.logaddexp(logS0, logSDiff+lZ)
        return logZstar     
        
    def likFunc_EP_asc_single(self, y, mu, s2, hyp):
        logS0, logSDiff, sDiff = hyp[1], hyp[2], hyp[3]
        likFunc_numer, likFunc_denom = hyp[5], hyp[6]
        lZ, dlZ, d2lZ = likFunc_numer(1, mu, s2, hyp)
        logZstar = np.logaddexp(logS0, logSDiff+lZ)
        expDiff = np.exp(lZ-logZstar)
        temp =  sDiff * expDiff
        dZstar  = temp * dlZ
        d2Zstar = temp * (d2lZ + dlZ**2 * (1-temp))
        
        return logZstar, dZstar, d2Zstar
        
        
    def likFunc_EP_both_single(self, y, mu, s2, hyp):       
        logS0, logSDiff, sDiff = hyp[1], hyp[2], hyp[3]
        likFunc_numer, likFunc_denom = hyp[5], hyp[6]
        lZ_numer, dlZ_numer, d2lZ_numer = likFunc_numer(y, mu, s2, hyp)
        lZ_numer += (logS0 if y<0 else 0)       
        lZ_denom, dlZ_denom, d2lZ_denom = likFunc_denom(y, mu, s2, hyp)
        return lZ_numer-lZ_denom, dlZ_numer-dlZ_denom, d2lZ_numer-d2lZ_denom
        
    def likFunc_EP_both_multi(self, y, mu, s2, hyp):
        logS0, logSDiff, sDiff = hyp[1], hyp[2], hyp[3]
        likFunc_numer, likFunc_denom = hyp[5], hyp[6]
        lZ_numer = likFunc_numer(y, mu, s2, hyp)        
        lZ_numer[y<0] += logS0                                      #note: we assume that logS1=0
        lZ_denom = likFunc_denom(y, mu, s2, hyp)
        return lZ_numer-lZ_denom
        
        
        
    def evalLL_EP(self, hyp):
        try: hyp[0]
        except: hyp=np.array([hyp])
        tol = 1e-4; max_sweep = 20; min_sweep = 2     #tolerance to stop EP iterations
        p = np.mean(self.y>0)
        s1 = 1.0
        s0 = s1 * self.prev / (1-self.prev) * (1-p) / p
        logS0 = np.log(s0); sDiff = s1-s0; logSDiff = np.log(sDiff)
        
        K = self.kernel.getTrainKernel(hyp)     
        m = np.zeros(self.y.shape[0])
        controls = (self.y < 0)
        cases = ~controls
        
        
        diagK = np.diag(K)
        sig2g = (1-self.prev)*np.mean(diagK[controls]) + self.prev*np.mean(diagK[cases])        
        if (sig2g > self.h2Scale): raise ValueError('sig2g larger than h2Scale found')       

        
        if (self.covars.shape[1] > 0):          
            C = self.covars
            logreg = sklearn.linear_model.LogisticRegression(penalty='l2', C=1000, fit_intercept=True)
            s0 = self.prev / (1-self.prev) * (1-np.mean(self.y>0)) / np.mean(self.y>0)
            logreg.fit(C, self.y)
            Pi = logreg.predict_proba(C)[:,1]
            Ki = Pi * s0 / (1 - Pi*(1-s0))
            if (self.eDist == 'logistic'):
                old_prev = self.prev
                t = np.empty(self.y.shape[0])
                for i in range(self.y.shape[0]):
                    self.prev = Ki[i]
                    t[i] = self.computeT(K, self.h2Scale-sig2g)
                self.prev = old_prev
            else: t = stats.norm(0,1).isf(Ki)
            

        if (self.eDist == 'normal'):            
            likFunc_numer_multi = self.likProbit_EP_multi
            likFunc_numer_single = self.likProbit_EP_single
            sig2e = self.h2Scale - sig2g
            if (self.covars.shape[1] == 0): t = np.zeros(self.y.shape[0]) + stats.norm(0, np.sqrt(sig2g+sig2e)).isf(self.prev)
            #t = stats.norm(0, np.sqrt(sig2g+sig2e)).isf(self.prev)
        elif (self.eDist == 'logistic'):                        
            likFunc_numer_multi = self.likLogistic_EP_multi_new
            likFunc_numer_single = self.likLogistic_EP_single_new
            sig2e = (self.h2Scale - sig2g) / (np.pi**2 / 3.0)
            #if (self.covars.shape[1] == 0): t = np.zeros(self.y.shape[0]) + self.computeT(K, self.h2Scale-sig2g)
            t = self.computeT(K, self.h2Scale-sig2g)
        else: raise ValueError('unknown eDist')
        
        likHyp_multi =  [sig2e, logS0, logSDiff, sDiff, t, likFunc_numer_multi, self.likFunc_EP_asc_multi]      
        likHyp_single = [sig2e, logS0, logSDiff, sDiff, t, likFunc_numer_single, self.likFunc_EP_asc_single]        
        likFuncMulti = likFunc_numer_multi
        likFuncSingle = likFunc_numer_single
        Sigma = K.copy()
        mu = m.copy() #- t
        
        nlZ0 = -np.sum(likFuncMulti(self.y, mu, np.diag(K), likHyp_multi))      
        ttau, tnu = np.zeros(self.y.shape[0]), np.zeros(self.y.shape[0])
        nlZ_old, sweep = np.inf, 0
        nlZ = nlZ0      
        while ((np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or sweep<min_sweep):
            nlZ_old = nlZ
            sweep+=1
            if (self.eDist == 'logistic'): ttau, tnu = self.EP_innerloop2(Sigma, self.y, mu, ttau, tnu, likFuncSingle, likHyp_single)
            else: ttau, tnu = ep_fast.EP_innerloop_probit(Sigma, self.y, mu, ttau, tnu, sig2e, t)
            (Sigma, mu, L, alpha, nlZ) = self.epComputeParams2(K, self.y, ttau, tnu, m, likFuncMulti, likHyp_multi)
        if (sweep == max_sweep and np.abs(nlZ-nlZ_old) > tol):
            nlZ = np.inf
        
        if (nlZ < 0): nlZ = np.inf
            
        self.mu = mu
            
        return nlZ
        
        
        
        
        


    def evalLL_AEP(self, hyp, grad=False, update_freq=1):
        try: hyp[0]
        except: hyp=np.array([hyp])
        tol = 1e-4; max_sweep = 20; min_sweep = 2     #tolerance to stop EP iterations
        p = np.mean(self.y>0)
        s1 = 1.0
        s0 = s1 * self.prev / (1-self.prev) * (1-p) / p
        
        y = self.y.copy()

        useCython = True
        
        
        logS0 = np.log(s0)
        sDiff = s1-s0
        logSDiff = np.log(sDiff)
    
        #Generate problem settings
        hyp_scaled = hyp.copy()
        if self.h2Scale != 1.0:
            hyp_scaled[-1] = np.log(np.exp(2*hyp[-1]) * self.h2Scale) / 2.0        
        
        K = self.kernel.getTrainKernel(hyp_scaled)        
        C = self.covars.copy()
        
        m = np.zeros(y.shape[0])       
        controls = (y < 0)
        cases = ~controls
        
        diagK = np.diag(K)
        sig2g = np.exp(2*hyp[-1])
        
        
        if (self.eDist == 'normal'): sig2e = self.h2Scale - sig2g
        elif (self.eDist == 'logistic'): sig2e = (self.h2Scale - sig2g) / (np.pi**2 / 3.0)
        else: raise ValueError('unknown eDist')
        if (sig2g > self.h2Scale):
            raise ValueError('sig2g larger than h2Scale found')
            

            
        if C.shape[1] > 0 and self.useFixed:
            logreg = sklearn.linear_model.LogisticRegression(penalty='l2', C=1000, fit_intercept=True)
            s0 = self.prev / (1-self.prev) * (1-np.mean(y>0)) / np.mean(y>0)
            logreg.fit(C, y)
            Pi = logreg.predict_proba(C)[:,1]
            Ki = Pi * s0 / (1 - Pi*(1-s0))          
            if (self.eDist == 'logistic'):
                old_prev = self.prev
                t = np.empty(y.shape[0])
                for i in range(y.shape[0]):
                    self.prev = Ki[i]
                    t[i] = self.computeT(K, self.h2Scale-sig2g)
                self.prev = old_prev
                
            else: t = stats.norm(0, np.sqrt(sig2g+sig2e)).isf(Ki)

        if (self.eDist == 'normal'):
            likFunc_numer_single = self.likProbit_EP_single
            likFunc_numer_multi = self.likProbit_EP_multi
            if (C.shape[1] == 0 or not self.useFixed): t = stats.norm(0, np.sqrt(sig2g+sig2e)).isf(self.prev)
        elif (self.eDist == 'logistic'):            
            likFunc_numer_single = self.likLogistic_EP_single_new
            likFunc_numer_multi = self.likLogistic_EP_multi_new
            if (C.shape[1] == 0 or not self.useFixed): t = self.computeT(K, self.h2Scale-sig2g)           
        else: raise ValueError('unknown eDist')
                
        likHyp_multi = [sig2e, logS0, logSDiff, sDiff, t, likFunc_numer_multi, self.likFunc_EP_asc_multi]
        likHyp_single = [sig2e, logS0, logSDiff, sDiff, t, likFunc_numer_single, self.likFunc_EP_asc_single]
        likFuncMulti = self.likFunc_EP_both_multi
        likFuncSingle = self.likFunc_EP_both_single
        
        #initialize Sigma and mu, the parameters of the Gaussian posterior approximation
        Sigma = K.copy()
        mu = m.copy()

        #marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*       
        nlZ0 = -np.sum(likFuncMulti(y, mu, np.diag(K), likHyp_multi))
        
        ttau, tnu = np.zeros(y.shape[0]), np.zeros(y.shape[0])
        nlZ_old, sweep = np.inf, 0
        nlZ = nlZ0
        
        while ((np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or sweep<min_sweep):
            nlZ_old = nlZ
            sweep+=1
            if (self.eDist == 'logistic' or not useCython): ttau, tnu = self.EP_innerloop2(Sigma, y, mu, ttau, tnu, likFuncSingle, likHyp_single)
            else:       
                ttau, tnu = ep_fast.EP_innerloop_probit_both_parallel(Sigma, y, mu, s0, sDiff, ttau, tnu, sig2e, np.zeros(y.shape[0])+t, update_freq=update_freq)
            try:
                (Sigma, mu, L, alpha, nlZ) = self.epComputeParams2(K, y, ttau, tnu, m, likFuncMulti, likHyp_multi)
            except:
                nlZ=np.inf
                print('\t', 'Cholesky failed!')
                raise
                break               
        if (sweep == max_sweep and np.abs(nlZ-nlZ_old) > tol):
            nlZ = np.inf
        nlZ_asc = nlZ
        
        if (len(self.prev_nlZ) >= 2):       
            prev_diff = np.maximum(np.abs(self.prev_nlZ[-1]-self.prev_nlZ[-2]), 2)
            bad_inds = ((np.abs(ttau)>100) | (np.abs(tnu)>100))
            if (np.abs(nlZ - self.prev_nlZ[-1]) > 2*np.abs(prev_diff) and np.any(bad_inds)):
                nlZ = np.inf
                nlZ_asc = nlZ
        
        if (nlZ == np.inf):
            self.old_ttau
            tol=1e-2
            ttau, tnu = self.old_ttau, self.old_tnu         
            
            Sigma = self.old_Sigma
            mu = self.old_mu
            nlZ_old, sweep = np.inf, 0
            nlZ = np.inf
            nlZ_arr = []
            max_sweep=40
            
            while (sweep<min_sweep or (np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep)):
                nlZ_old = nlZ
                sweep+=1
                if (self.eDist == 'logistic' or not useCython): ttau, tnu = self.EP_innerloop2(Sigma, y, mu, ttau, tnu, likFuncSingle, likHyp_single)
                else:
                    ttau, tnu = ep_fast.EP_innerloop_probit_both_parallel(Sigma, y, mu, s0, sDiff, ttau, tnu, sig2e, np.zeros(y.shape[0])+t, update_freq=update_freq)
                try:
                    (Sigma, mu, L, alpha, nlZ) = self.epComputeParams2(K, y, ttau, tnu, m, likFuncMulti, likHyp_multi)
                except:
                    nlZ = np.inf
                    break
                nlZ_arr.append(nlZ)
            nlZ_arr = np.array(nlZ_arr)
            if (sweep == max_sweep and np.abs(nlZ-nlZ_old) > tol):
                if (np.abs(nlZ-nlZ_old) < 3):
                    if (np.all(nlZ_arr[5:] < self.old_nlZ)): nlZ = np.max(nlZ_arr[5:])
                    elif (np.all(nlZ_arr[5:] > self.old_nlZ)): nlZ = np.min(nlZ_arr[5:])
                else:
                    nlZ = np.inf
            prev_diff = np.maximum(np.abs(self.prev_nlZ[-1]-self.prev_nlZ[-2]), 2)
            bad_inds = ((np.abs(ttau)>100) | (np.abs(tnu)>100))         
            
            try:
                if (nlZ < np.inf and np.max(np.abs(nlZ_arr[5:] - self.prev_nlZ[-1])) > 2*np.abs(prev_diff) and np.any(bad_inds)):
                    nlZ = np.inf    
            except:
                pass

            nlZ_asc = nlZ           
        
            
        if (nlZ < np.inf):
            self.old_ttau, self.old_tnu, self.old_Sigma, self.old_mu, self.old_nlZ = ttau, tnu, Sigma, mu, nlZ
            self.prev_nlZ.append(nlZ)
            self.mu = mu
                
                        
        nlZ = nlZ_asc
        if (nlZ < 0): nlZ = np.inf

        return nlZ
        
        
    def likLogistic_EP_multi(self, y, mu, s2, hyp=None):
    
        lZc = self.likErf_EP(np.outer(y, np.ones(5)), np.outer(mu, self.logistic_lam), np.outer(s2, self.logistic_lam2), compDerivs=False)
        lZ = self.log_expA_x_multi(lZc, self.logistic_c)                            #A=lZc, B=dlZc, d=c.*lam', lZ=log(exp(A)*c)
        
        val = np.abs(mu) - 196/200*s2-4                         #empirically determined bound at val==0
        lam = 1.0 / (1.0+np.exp(-10*val))                       #interpolation weights
        lZtail = np.minimum(s2/2.0-np.abs(mu), -0.1)            #apply the same to p(y|f) = 1 - p(-y|f)
        muy = mu*y
        id = muy>0; lZtail[id] = np.log(1-np.exp(lZtail[id]))   #label and mean agree
        lZ   = (1-lam)*lZ + lam*lZtail                          #interpolate between scale mixture and tail approximation
        return lZ
        
        
    #computes y = log( exp(A)*x ) in a numerically safe way by subtracting the
    # maximal value in each row to avoid cancelation after taking the exp
    def log_expA_x_multi(self, A, x):       
        maxA = np.max(A, axis=1)                                    #number of columns, max over columns        
        y = np.log(np.exp(A - maxA[:, np.newaxis]).dot(x)) + maxA   #exp(A) = exp(A-max(A))*exp(max(A))
        return y
        
    #computes y = log( exp(A)*x ) in a numerically safe way by subtracting the
    # maximal value in each row to avoid cancelation after taking the exp
    def log_expA_x_single(self, A, x):      
        maxA = np.max(A)                            #number of columns, max over columns        
        y = np.log(np.exp(A-maxA).dot(x)) + maxA    #exp(A) = exp(A-max(A))*exp(max(A))
        return y
        
    # computes y = ( (exp(A).*B)*z ) ./ ( exp(A)*x ) in a numerically safe way.
    #The function is not general in the sense that it yields correct values for
    #all types of inputs. We assume that the values are close together.
    def expABz_expAx_single(self, A,x,B,z):
        maxA = np.max(A)                        #number of columns, max over columns        
        expA = np.exp(A-maxA)
        y = np.dot(expA*B, z) / np.dot(expA, x)
        return y


        
        
        
        
    def evalLL(self, hyp, method):
        if (method == 'aep'): return self.evalLL_AEP(hyp)
        elif (method == 'aep_parallel'): return self.evalLL_AEP(hyp, update_freq=10000000000)
        elif (method == 'ep'): return self.evalLL_EP(hyp)       
        else: raise ValueError('unrecognized method: %s. Valid methods are reml, pcgc, apl, aep, aep_parallel or ep'%(method))
        
        
        
    def reml(self, is_binary):
        K = self.kernel.getTrainKernel(np.array([0]))
        logdetXX = 0
        
        #eigendecompose
        s,U = la.eigh(K)
        s[s<0]=0    
        ind = np.argsort(s)[::-1]
        U = U[:, ind]
        s = s[ind]  
        
        #Prepare required matrices
        if is_binary: y = (self.y>0).astype(np.int)
        else: y = self.y_cont
        Uy = U.T.dot(y).flatten()
        covars = np.ones((y.shape[0], 1))
        UX = U.T.dot(covars)        
        
        if (U.shape[1] < U.shape[0]):
            UUX = covars - U.dot(UX)
            UUy = y - U.dot(Uy)
            UUXUUX = UUX.T.dot(UUX)
            UUXUUy = UUX.T.dot(UUy)
            UUyUUy = UUy.T.dot(UUy)
        else: UUXUUX, UUXUUy, UUyUUy = None, None, None
        n = U.shape[0]
        ldeltaopt_glob = optimize.minimize_scalar(self.negLLevalLong, bounds=(-5, 5), method='Bounded', args=(s, Uy, UX, logdetXX, UUXUUX, UUXUUy, UUyUUy, n)).x
        
        ll, sig2g, beta, r2 = self.negLLevalLong(ldeltaopt_glob, s, Uy, UX, logdetXX, UUXUUX, UUXUUy, UUyUUy, n, returnAllParams=True)
        sig2e = np.exp(ldeltaopt_glob) * sig2g
                
        return sig2g/(sig2g+sig2e)
        
        
    def negLLevalLong(self, logdelta, s, Uy, UX, logdetXX, UUXUUX, UUXUUy, UUyUUy, numIndividuals, returnAllParams=False):
        Sd = s + np.exp(logdelta)
        UyS = Uy / Sd
        yKy = UyS.T.dot(Uy)
        logdetK = np.log(Sd).sum()
        null_ll, sigma2, beta, r2 = self.lleval(Uy, UX, Sd, yKy, logdetK, logdetXX, logdelta, UUXUUX, UUXUUy, UUyUUy, numIndividuals)
        if returnAllParams: return null_ll, sigma2, beta, r2
        else: return -null_ll
        
        
    def lleval(self, Uy, UX, Sd, yKy, logdetK, logdetXX, logdelta, UUXUUX, UUXUUy, UUyUUy, numIndividuals):
        N = numIndividuals
        D = UX.shape[1]
                
        UXS = UX / np.lib.stride_tricks.as_strided(Sd, (Sd.size, D), (Sd.itemsize,0))
        XKy = UXS.T.dot(Uy)         
        XKX = UXS.T.dot(UX) 
        
        if (Sd.shape[0] < numIndividuals):
            delta = np.exp(logdelta)
            denom = delta
            XKX += UUXUUX / denom
            XKy += UUXUUy / denom
            yKy += UUyUUy / denom           
            logdetK += (numIndividuals-Sd.shape[0]) * logdelta      
            
        [SxKx,UxKx]= la.eigh(XKX)   
        i_pos = SxKx>1E-10
        beta = np.dot(UxKx[:,i_pos], (np.dot(UxKx[:,i_pos].T, XKy) / SxKx[i_pos]))
        r2 = yKy-XKy.dot(beta)

        reml = True
        if reml:
            logdetXKX = np.log(SxKx).sum()
            sigma2 = (r2 / (N - D))
            ll =  -0.5 * (logdetK + (N-D)*np.log(2.0*np.pi*sigma2) + (N-D) + logdetXKX - logdetXX)
        else:
            sigma2 = r2 / N
            ll =  -0.5 * (logdetK + N*np.log(2.0*np.pi*sigma2) + N)
            
        return ll, sigma2, beta, r2
        
    def solveChol(self, L, B, overwrite_b=True):
        cholSolve1 = la.solve_triangular(L, B, trans=1, check_finite=False, overwrite_b=overwrite_b)
        cholSolve2 = la.solve_triangular(L, cholSolve1, check_finite=False, overwrite_b=True)
        return cholSolve2

        
    def evalLL_EP(self, hyp):
        tol = 1e-4; max_sweep = 20; min_sweep = 2     #tolerance to stop EP iterations      
        s0 = self.prev / (1-self.prev)
        s1 = 1.0
        useCython = False
        try: hyp[0]
        except: hyp=np.array([hyp])
        
        if (self.prev < 0.5):
            logS0 = np.log(s0)
            logSdiff = np.log(s1-s0)
        else:
            logS0 = -np.inf
            logSdiff = 0.0
    
        #Generate problem settings
        K = self.kernel.getTrainKernel(hyp)
        m = np.zeros(self.y.shape[0])
        if self.useFixed: m += self.covars.dot(self.fixedEffects)
        controls = (self.y < 0)
        cases = ~controls       
        diagK = np.diag(K)      
        sig2g = (1-self.prev)*np.mean(diagK[controls]) + self.prev*np.mean(diagK[cases])
        if (sig2g > 1.0): raise ValueError('sig2g larger than 1.0 found')
        sig2e = 1.0 - sig2g
        t = stats.norm(0, np.sqrt(sig2g+sig2e)).isf(self.prev)
        m -= t

        if useCython:
            EP_func = EP_cython.EPInnerLoop_cython
        else:
            EP_func = self.EPInnerLoop
            
        llFunc = self.llFuncStandard
        
        #A note on naming (taken directly from the GPML documentation):
        #variables are given short but descriptive names in 
        #accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
        #and s2 are mean and variance, nu and tau are natural parameters. A leading t
        #means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
        #for a vector of cavity parameters. N(f|mu,Sigma) is the posterior.
        
        #initialize Sigma and mu, the parameters of the Gaussian posterior approximation
        Sigma = K.copy()
        mu = m.copy()
        
        #marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*       
        nlZ0 = -np.sum(llFunc(self.y, mu, np.diag(K), sig2e))
        
        ttau, tnu = np.zeros(self.y.shape[0]), np.zeros(self.y.shape[0])
        nlZ_old, sweep = np.inf, 0
        nlZ = nlZ0
        
        while ((np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or sweep<min_sweep):
            nlZ_old = nlZ
            sweep+=1
            Sigma, mu, ttau, tnu = EP_func(Sigma, self.y, mu, ttau, tnu, sig2e)
                
            #recompute since repeated rank-one updates can destroy numerical precision
            (Sigma, mu, L, alpha, nlZ) = self.epComputeParams(K, self.y, ttau, tnu, sig2e, m, llFunc)

        self.mu = mu
        return nlZ
        
        
        
    def llFuncStandard(self, y, mu, s2, sig2e):
        z = mu / np.sqrt(sig2e+s2) * y
        nlZ = stats.norm(0,1).logcdf(z)
        return nlZ
        
        
        
    def EP_innerloop2(self, Sigma, y, mu, ttau, tnu, likFuncSingle, likHyp):
    
        randpermN = np.random.permutation(range(y.shape[0]))
        normPDF = stats.norm(0,1)
        
        for i in randpermN:     #iterate EP updates (in random order) over examples
        
            #first find the cavity distribution params tau_ni and nu_ni
            
            if (ttau[i] > 1.0/Sigma[i,i]):
                raise ValueError('infeasible ttau[i] found!!!')

                
            tau_ni = 1.0/Sigma[i,i]  - ttau[i]              #Equation 3.56 rhs (and 3.66) from GP book
            nu_ni = (mu[i]/Sigma[i,i] - tnu[i])             #Equation 3.56 lhs (and 3.66) from GP book
            mu_ni = nu_ni / tau_ni
            
            #compute the desired derivatives of the individual log partition function
            try:
                t = likHyp[4]               
                likHyp[4] = t[i]
                lZ, dlZ, d2lZ = likFuncSingle(y[i], mu_ni, 1.0/tau_ni, likHyp)
                likHyp[4] = t
            except:
                lZ, dlZ, d2lZ = likFuncSingle(y[i], mu_ni, 1.0/tau_ni, likHyp)
            
            
            ttau_old, tnu_old = ttau[i], tnu[i]                     #find the new tilde params, keep old
            ttau[i] = -d2lZ  / (1+d2lZ/tau_ni)          
            ttau[i] = np.maximum(ttau[i], 0)                        #enforce positivity i.e. lower bound ttau by zero
            tnu[i]  = (dlZ - mu_ni*d2lZ ) / (1+d2lZ/tau_ni)
            if (ttau[i] == 0): tnu[i]=0
            
            dtt = ttau[i] - ttau_old
            dtn = tnu[i] - tnu_old      #rank-1 update Sigma
            si = Sigma[:,i]
            ci = dtt / (1+dtt*si[i])
            
            mu -= (ci* (mu[i]+si[i]*dtn) - dtn) * si                    #Equation 3.53 from GP book
            Sigma -= np.outer(ci*si, si)                                #Equation 3.70 from GP book (#takes 70% of total time)
        return  ttau, tnu
        
        
    def EPInnerLoop(self, Sigma, y, mu, ttau, tnu, sig2e):
    
        randpermN = np.random.permutation(range(y.shape[0]))
        normPDF = stats.norm(0,1)
                
        for i in randpermN:     #iterate EP updates (in random order) over examples
        
            #first find the cavity distribution params tau_ni and mu_ni
            tau_ni = 1.0/Sigma[i,i]  - ttau[i]              #Equation 3.56 rhs (and 3.66) from GP book
            mu_ni = (mu[i]/Sigma[i,i] - tnu[i]) / tau_ni    #Equation 3.56 lhs (and 3.66) from GP book

            #compute the desired derivatives of the individual log partition function
            s2 = 1.0/tau_ni
            sqrtS2 = np.sqrt(s2 + sig2e)
            z = mu_ni * y[i] / sqrtS2                               #Equation 3.82 from GP book
            ttau_old, tnu_old = ttau[i], tnu[i]                     #find the new tilde params, keep old
            Z = normPDF.logcdf(z)
            n_p = np.exp(normPDF.logpdf(z) - Z)                     #Equation 3.82 from GP book

            #matlab computation...          
            dlZ = y[i] * n_p / sqrtS2                   #1st derivative of log(Z) wrt mean
            d2lZ = -n_p*(z+n_p)/(sig2e+s2)          #2nd derivative of log(Z) wrt mean
            ttau_matlab = -d2lZ  / (1+d2lZ/tau_ni)          
            tnu_matlab  = (dlZ - mu_ni*d2lZ ) / (1+d2lZ/tau_ni)
            
            #my new computation...
            meanQx = mu_ni + s2*n_p * y[i] / sqrtS2             #This is mu_hat from Equations 3.57-3.59 (specifically this is Equation 3.85)
            meanQx2 = dlZ/tau_ni + mu_ni
            assert np.isclose(meanQx, meanQx2)
            varQx = s2 - s2**2 * n_p / (sig2e+s2) * (z + n_p)   #This is sigma^2_hat from Equations 3.57-3.59 (specifically this is equation 3.87)
            #varQx2 = d2lZ/tau_ni**2 + 2*mu_ni*meanQx - mu_ni**2 + 1.0/tau_ni + dlZ**2/tau_ni**2 - meanQx2**2
            varQx2 = (d2lZ+dlZ**2)/tau_ni**2 + 2*mu_ni*meanQx - mu_ni**2 + 1.0/tau_ni - meanQx2**2
            assert np.isclose(varQx, varQx2)
            ttau[i] = 1.0/varQx - tau_ni                        #Equation 3.59 (and 3.66)
            tnu[i] = meanQx/varQx - mu_ni*tau_ni                #Equation 3.59 (and 3.66)
            
            ttau[i] = np.maximum(ttau[i], 0)        #enforce positivity i.e. lower bound ttau by zero
            dtt = ttau[i] - ttau_old
            dtn = tnu[i] - tnu_old      #rank-1 update Sigma
            si = Sigma[:,i]
            ci = dtt / (1+dtt*si[i])
            
            mu -= (ci* (mu[i]+si[i]*dtn) - dtn) * si                    #Equation 3.53 from GP book
            Sigma -= np.outer(ci*si, si)                                #Equation 3.70 from GP book (#takes 70% of total time)
            
        return  Sigma, mu, ttau, tnu
        
        
        
        
        

        
        
        
    def epComputeParams2(self, K, y, ttau, tnu, m, likFuncMulti, likHyp):
    
        n = y.shape[0]
        sW = np.sqrt(ttau)                                   #compute Sigma and mu  
        L = la.cholesky(np.eye(n) + np.outer(sW, sW) * K, overwrite_a=True, check_finite=False)
        
        #L.T*L=B=eye(n)+sW*K*sW 
        V = la.solve_triangular(L, K*np.tile(sW, (n, 1)).T, trans=1, check_finite=False, overwrite_b=True)
        Sigma = K - V.T.dot(V)
        alpha = tnu-sW * self.solveChol(L, sW*(K.dot(tnu)+m))
        mu = K.dot(alpha) + m
        v = np.diag(Sigma)
        
        tau_n = 1.0/np.diag(Sigma) - ttau                           #compute the log marginal likelihood
        nu_n  = mu/np.diag(Sigma) - tnu                             #vectors of cavity parameters
        
        lZ = likFuncMulti(y, nu_n/tau_n, 1.0/tau_n, likHyp)
        p = tnu - m*ttau                                            #auxiliary vectors
        q = nu_n - m*tau_n                                          #auxiliary vectors
        nlZ = (np.sum(np.log(np.diag(L))) - lZ.sum() - (p.T.dot(Sigma)).dot(p/2.0) + (v.T.dot(p**2))/2.0 
        - q.T.dot((ttau/tau_n*q - 2*p) * v)/2.0 - np.sum(np.log(1+ttau/tau_n))/2.0)
        
        return (Sigma, mu, L, alpha, nlZ)
        
        
    def epComputeParams(self, K, y, ttau, tnu, sig2e, m, llFunc):
        n = y.shape[0]
        sW = np.sqrt(ttau)                                   #compute Sigma and mu  
        L = la.cholesky(np.eye(n) + np.outer(sW, sW) * K, overwrite_a=True, check_finite=False)
        
        #L.T*L=B=eye(n)+sW*K*sW 
        V = la.solve_triangular(L, K*np.tile(sW, (n, 1)).T, trans=1, check_finite=False, overwrite_b=True)
        Sigma = K - V.T.dot(V)
        alpha = tnu-sW * self.solveChol(L, sW*(K.dot(tnu)+m))
        mu = K.dot(alpha) + m
        v = np.diag(Sigma)
        
        tau_n = 1.0/np.diag(Sigma) - ttau                           #compute the log marginal likelihood
        nu_n  = mu/np.diag(Sigma) - tnu                             #vectors of cavity parameters
        
        mu_temp = nu_n/tau_n
        s2 = 1.0/tau_n
        lZ = llFunc(y, mu_temp, s2, sig2e)
        p = tnu - m*ttau                                            #auxiliary vectors
        q = nu_n - m*tau_n                                          #auxiliary vectors
        nlZ = (np.sum(np.log(np.diag(L))) - np.sum(lZ) - (p.T.dot(Sigma)).dot(p/2.0) + (v.T.dot(p**2))/2.0 
        - q.T.dot((ttau/tau_n*q - 2*p) * v)/2.0 - np.sum(np.log(1+ttau/tau_n))/2.0)
        
        return (Sigma, mu, L, alpha, nlZ)
        
        
    def solveChol(self, L, B, overwrite_b=True):
        cholSolve1 = la.solve_triangular(L, B, trans=1, check_finite=False, overwrite_b=overwrite_b)
        cholSolve2 = la.solve_triangular(L, cholSolve1, check_finite=False, overwrite_b=True)
        return cholSolve2
        
        
        
        
    
    def pairwise_ml(self):
        
        K = self.kernel.getTrainKernel(np.array([0]))
            
        yBinary = (self.y>0).astype(np.int)
        t = stats.norm(0,1).isf(self.prev)
        
        #estimate initial fixed effects
        C = self.covars
        if C.shape[1] > 0 and self.useFixed:
            logreg = sklearn.linear_model.LogisticRegression(penalty='l2', C=1000, fit_intercept=True)
            s0 = self.prev / (1-self.prev) * (1-np.mean(yBinary>0)) / np.mean(yBinary>0)
            logreg.fit(C, yBinary)
            Pi = logreg.predict_proba(C)[:,1]
            Ki = Pi * s0 / (1 - Pi*(1-s0))
            t = stats.norm(0,1).isf(Ki)
                    
        
        
        phit = stats.norm(0,1).pdf(t)
        ysum_temp = np.tile(yBinary, (yBinary.shape[0], 1))
        sumY = ysum_temp + ysum_temp.T
        #sumY_flat = sumY[np.triu_indices(K.shape[0], 1)]
        Y0 = (sumY==0)
        Y1 = (sumY==1)
        Y2 = (sumY==2)
        
        P = np.mean(yBinary)
        denom = (self.prev**2 * (1-self.prev)**2)
        coef0 = phit**2 * P * (1-P)**2 * (2*self.prev-P) / denom
        coef1 = -(phit**2 * 2 * P * (1-P) * (P**2 + self.prev - 2*self.prev*P)) / denom
        coef2 = phit**2 * (1-P) * P**2 * (1-2*self.prev+P) / denom
        intercept = Y0*(1-P)**2 + Y1*2*P*(1-P) + Y2*P**2
        
        coef = Y0*coef0 + Y1*coef1 + Y2*coef2
        coefG = coef*K
        np.fill_diagonal(coefG, 0)      #to ensure log(intercept + coefG*h2)=0 in diagonal
        np.fill_diagonal(intercept, 1)  #to ensure log(intercept + coefG*h2)=0 in diagonal        
        
        def pw_nll(h2):
            ll = np.sum(np.log(intercept + coefG*h2))
            if np.isnan(ll): ll=-np.inf
            return -ll
        
        optObj = optimize.minimize_scalar(pw_nll, bounds=(0, 1), method='bounded')     
        best_h2 = optObj.x
        return best_h2, optObj.fun
    
    
    
    def pcgc(self, rbf_hyp=None):
        t = stats.norm(0,1).isf(self.prev)
        if rbf_hyp is None:
            K = self.kernel.getTrainKernel(np.array([0]))
        else:
            K = self.kernel.getTrainKernel(np.array([rbf_hyp, 0]))
        y = self.y.copy()
        y[y>0] = 1
        y[y<=0] = 0
        C = self.covars
        
        if rbf_hyp is None and (C.shape[1] == 0 or not self.useFixed) and False:
            P = np.sum(y>0) / float(y.shape[0])     
            phit = stats.norm(0,1).pdf(t)
            xCoeff = P*(1-P) / (self.prev**2 * (1-self.prev)**2) * phit**2
            yBinary = (y>0).astype(np.int)
            yy = np.outer((yBinary-P) / np.sqrt(P*(1-P)), (yBinary-P) / np.sqrt(P*(1-P)))           
            xx = xCoeff * K

            yy = yy[np.triu_indices(yy.shape[0], 1)]
            xx = xx[np.triu_indices(xx.shape[0], 1)]
            
            slope, intercept, rValue, pValue, stdErr = stats.linregress(xx,yy)
            return slope, 0
            
        #estimate initial fixed effects
        if (C.shape[1] > 0):
            logreg = sklearn.linear_model.LogisticRegression(penalty='l2', C=1000, fit_intercept=True)
            s0 = self.prev / (1-self.prev) * (1-np.mean(y>0)) / np.mean(y>0)
            logreg.fit(C, y)
            Pi = logreg.predict_proba(C)[:,1]
            Ki = Pi * s0 / (1 - Pi*(1-s0))
            t_i = stats.norm(0,1).isf(Ki)
        else:
            t_i = t

        
        np.fill_diagonal(K, 0)
        s0 = self.prev / (1-self.prev) * (1-np.mean(y>0)) / np.mean(y>0)
        gradsFields = GradientFields(K, s0, t_i, self.prev)
        
        #Compute the elements of the function value (the squared distance between the observed and expected pairwise phenotypic covariance)
        K_nodiag_AB0_norm2 = np.sum(gradsFields.K_nodiag_sqr_AB0 * gradsFields.Atag0/gradsFields.B0)
        z = (y - gradsFields.Pi) / gradsFields.stdY
        z_K_nodiag_AB0_z = z.dot(gradsFields.K_nodiag_AB0).dot(z)
        
        #compute h2
        h2 = z_K_nodiag_AB0_z / K_nodiag_AB0_norm2
        
        #compute function value        
        zTz = z.dot(z)
        loss = K_nodiag_AB0_norm2 - 2*z_K_nodiag_AB0_z + zTz**2
        z2 = z**2
        loss -= z2.dot(z2)
        
        return h2, loss

            
            
            
def estimate_params(prevTest, method, prev, kernel, sig2gList, rbf_scale_list, opt_sigma2g=False):
    
    #pairwise-ML
    if method == 'apl':
        assert kernel == 'linear'
        pw_loss_list = []
        pw_h2_list = []
        sigma2_est, _ = prevTest.pairwise_ml()
        rbf_est = None

    #PCGC
    elif method == 'pcgc':
        if kernel == 'linear':
            rbf_hyp_list = [None]
        else:
            rbf_hyp_list = rbf_scale_list
        pcgc_loss_list = []
        pcgc_h2_list = []
        for rbf_hyp in rbf_hyp_list:
            pcgc_h2, pcgc_loss = prevTest.pcgc(rbf_hyp=(None if rbf_hyp is None else np.log(rbf_hyp)))
            pcgc_h2_list.append(pcgc_h2)
            pcgc_loss_list.append(pcgc_loss)
        sigma2_est = pcgc_h2_list[np.argmin(pcgc_loss_list)]
        rbf_est = rbf_hyp_list[np.argmin(pcgc_loss_list)]
        
    #REML
    elif method == 'reml':
        assert kernel == 'linear'
        h2Estimate = prevTest.reml(is_binary=True)
        t = stats.norm(0,1).isf(prev)
        P = np.mean(prevTest.y>0)
        coeff = (prev*(1-prev))**2 / (P*(1-P)) / stats.norm(0,1).pdf(t)**2
        sigma2_est = h2Estimate*coeff
        rbf_est = None

    #EP-based methods
    else:                   
        nllArr = np.zeros(len(sig2gList)) + np.inf
        for sig2g_i, sig2g in enumerate(sig2gList):
            hyp = np.log(sig2g)/2.0
            if kernel == 'rbf':
                hyp = np.array([np.log(rbf_scale_list[sig2g_i]), hyp])
            nlZ = prevTest.evalLL(hyp, method)
            nllArr[sig2g_i] = nlZ
            if (method == 'aep' and nlZ == np.inf):
                continue
            best_ind = np.argmin(nllArr)
        sigma2_est = sig2gList[best_ind]
        if kernel == 'rbf':
            rbf_est = rbf_scale_list[best_ind]
        else:
            rbf_est= None
        
        #run optimization algorithm if requested
        if opt_sigma2g:
            assert args.kernel == 'linear'
            bucketsSortedInd = np.argsort(nllArr)
            bestPoints = sorted([sig2gList[bucketsSortedInd[0]], sig2gList[bucketsSortedInd[1]], sig2gList[bucketsSortedInd[2]]])
            if (args.e_dist == 'normal'):
                lb = sig2gList[bucketsSortedInd[0]]-0.05
                ub = sig2gList[bucketsSortedInd[0]]+0.05
            else:
                lb = sig2gList[bucketsSortedInd[0]]-0.5
                ub = sig2gList[bucketsSortedInd[0]]+0.5
            
            if (lb<=0): lb = 1e-3
            if (ub<=0): ub = 1e-3
            lb = np.log(lb) / 2.0
            ub = np.log(ub) / 2.0
            optObj = optimize.minimize_scalar(lambda hyp: prevTest.evalLL(hyp, args.method), bounds=(lb, ub), method='bounded')
            sigma2_est = np.exp(2*optObj.x)        
        
    return sigma2_est, rbf_est

        
        
def main(args):
    #set random seed
    prng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    
    #determine sig2g grid
    if args.dense_grid:
        sig2gList = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,0.11,0.12,0.13,0.14, 0.15,0.16,0.17,0.18,0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60]
    else:
        sig2gList = [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    if (args.h2 is not None and args.h2 > 0.4): sig2gList += [0.6, 0.7, 0.8, 0.9]     
    
    #determine rbf scale grid
    if args.kernel=='rbf':
        num_rbf_params = 3
        rbf_scale_list = np.array(np.sort(np.unique(list(np.linspace(0.25, 0.75, num=num_rbf_params)) + [args.rbf_scale])))
        if args.method == 'pcgc':
            rbf_scale_list = np.array([args.rbf_scale])
        if 'pcgc' not in args.method and 'apl' not in args.method:
            rbf_scale_list = np.repeat(rbf_scale_list, len(sig2gList))
            sig2gList = np.tile(sig2gList, num_rbf_params)
    else:
        rbf_scale_list = None
            
    
    #prevTest is an object holding the generated data 
    prevTest = None
            
        
    #create the results arrays
    sig2gArr = np.empty(args.r)
    rbfscale_arr = np.empty(args.r)

    #iterate over experiments
    for r_i in range(args.r):
        
        #generate a dataset
        if prevTest is None or not args.no_regenerate:
            prevTest = PrevTest(args.n, args.m, args.prev, args.use_fixed, args.h2_scale, prng=prng, num_generate=args.n_generate)
        prevTest.genData(args.h2, args.e_dist, args.num_fixed, not args.no_ascertain, args.scale_g, extraSNPs=args.num_extra_snps, fixedVar=args.fixed_var, frac_cases=args.frac_cases, kernel=args.kernel, rbf_scale=args.rbf_scale)
        prevTest.prev_nlZ = []
        
        #estimate parameters
        sigma2_est, rbf_est = estimate_params(prevTest, args.method, args.prev, args.kernel, sig2gList, rbf_scale_list, opt_sigma2g=args.opt_sigma2g)
        sig2gArr[r_i] = sigma2_est
        rbfscale_arr[r_i] = rbf_est
            
        #print summary of all the results obtained so far
        print('%d mean sig2g-hat: %0.4f (%0.4f)'%(r_i+1, sig2gArr[:r_i+1].mean(), sig2gArr[:r_i+1].std()), end=' ')
        if args.kernel == 'rbf':
            print('%d mean rbf-hat: %0.4f (%0.4f)'%(r_i+1, rbfscale_arr[:r_i+1].mean(), rbfscale_arr[:r_i+1].std()), end=' ')
        print()
        

        
        
        
        
        
####################################################################################################################        
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500, help='Sample size')
    parser.add_argument('--m', type=int, default=500, help='Number of SNPs (or rank of linear covariance matrix)')
    parser.add_argument('--prev', type=float, default=0.01, help='Trait prevalence in the population')
    parser.add_argument('--h2', type=float, default=0.25, help='Heritability (or proportion of liability variance explained by the m variants')
    parser.add_argument('--r',type=int, default=100, help='Number of experiments to perform')
    parser.add_argument('--method', default='aep_parallel', help='Estimation method')
    parser.add_argument('--seed', type=int, default=3317, help='random seed')
    parser.add_argument('--opt_sigma2g', default=False, action='store_true', help='If set, run an optimization algorithm between the two best grid search points')
    parser.add_argument('--e_dist', default='normal', help='Distribution of the environmetal effect e')
    
    parser.add_argument('--num_fixed', type=int, default=0, help='number of covariates with fixed effects')
    parser.add_argument('--use_fixed', default=False, action='store_true', help='whether to directly estimate fixed effects')
    parser.add_argument('--no_regenerate',  default=False, action='store_true', help='If set, we will reuse the same data for all experiments to save time')
    parser.add_argument('--no_ascertain', default=False, action='store_true', help='If set, the data will not be ascertained')
    parser.add_argument('--scale_g', default=False, action='store_true', help='whether to scale g to ensure liability variance of 1.0')
    parser.add_argument('--frac_cases', type=float, default=0.5, help='fraction of cases in each sampled data set')
    parser.add_argument('--h2_scale', type=float, default=1.0, help='Liability scale (default: 1.0)')
    parser.add_argument('--fixed_var', type=float, default=0.5, help='variance explained by fixed effects (compared to sig2e=1.0)')
    parser.add_argument('--num_extra_snps', type=int, default=0, help='number of non-causal SNPs to add')
    parser.add_argument('--dense_grid', default=False, action='store_true', help='if determined, we will use a dense grid of sigma2g points')
    parser.add_argument('--kernel', default='linear', help='The kernel to use (either linear or rbf are currently supported)')
    parser.add_argument('--rbf_scale', type=float, default=0.5, help='the scale parameter of the RBF kernel')
    parser.add_argument('--n_generate', type=int, default=None, help='number of individuals to actually generate (if smaller than effcetive population size, we will create identical twins')
    args = parser.parse_args()
    
    main(args)
