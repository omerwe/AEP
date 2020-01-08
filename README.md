# AEP
Maximum Likelihood for Gaussian Process Classifiers under Case-Control Sampling

This page contains the code of the methods **AEP** (ascertained expectation propagation) and **APL** (ascertained pairwise likelihood) 
for fitting Gaussian processes (or generalized linear mixed models) under case-control ascertainment. The methods are described in 
[Weissbrod, Kaufman, Golan and Rosset 2019 JMLR](http://jmlr.org/papers/v20/18-298.html).

Parts of the code are loosely based on code translated from the [GPML toolbox](http://www.gaussianprocess.org/gpml/code/matlab).

<br>
<br>


# Installation
The code is designed for Python >=3.6 and requires the following freely available Python packages:
* [numpy](http://www.numpy.org/) and [scipy](http://www.scipy.org)
* [scikit-learn](http://scikit-learn.org/stable)
* [cython](https://github.com/cython/cython)

We recommend runningthe code using the [Anaconda Python distribution](https://www.anaconda.com/download/).
In Anaconda, you can install all the Python packages with the command "conda install \<package_name\>".
Alternatively, the Python packages can be installed with the command "pip install --user \<package_name\>".

Once all the prerequisite packages are installed, you can install the code on a git-enabled machine by typing:
```
git clone https://github.com/omerwe/AEP
cd AEP
python setup.py build_ext --inplace
```
The last command will compile some cython code. Please make sure that it completes without any error messages.


<br>

# Use instructions
You can invoke the code and run an experiment using the file `aep.py`. This script will simulate ascertained case-control data and will then estimate the model parameters using one of the methods described in the paper (see details below). Here's a tl;dr example:
```
python aep.py --r 100 --n 500 --frac_cases 0.5 --prev 0.01 --kernel linear --e_dist normal --m 1000 --h2 0.25 --method aep_parallel 
```
This will generate r=100 datasets, each consisting of n=500 individuals, 50% of which are cases and 50% are controls, assuming that the prevalence of cases in the general population is prev=1%. Each dataset will include m=1000 features used to determine the latent variable *g* (e.g. genetic variants) via a linear kernel with a normally-distributed likelihood (i.e. a probit link function). The latent variable *g* will explain 25% of the variance of the underlying liability. The code will try to estimate this variance (i.e. the scale parameter of the linear kernel) using the method **aep_parallel**. The first few lines of output should look like:
```
1 mean sig2g-hat: 0.2000 (0.0000)
2 mean sig2g-hat: 0.2000 (0.0000)
3 mean sig2g-hat: 0.2167 (0.0236)
4 mean sig2g-hat: 0.2250 (0.0250)
5 mean sig2g-hat: 0.2500 (0.0548)
6 mean sig2g-hat: 0.2583 (0.0534)
```
This shows the empirical average and standard deviation of the estimates of the scale parameter, which is updated after running each experiment. For example, after running six experiments the mean estimate was 0.2583 with stdev=0.0534.

<br>


## Script arguments ##
You can see all the arguments of `aep.py` by typing `python aep.py --help`. Here we describe the most important arguments:
<br><br>
`--method`: The method used to estimate the parameters. The supported methods are `ep` (expectation propagation), `aep` (ascertained expectation propagation), `aep_parallel` (ascertained expectation propagation with parallel updates), `apl` (ascertained pairwise likelihood), `pcgc` (phenotype-correlation-genotype-correlation), and `reml` (restricted maximum likelihood, using the correction of [Lee et al. 2011 AJHG](https://www.cell.com/ajhg/fulltext/S0002-9297(11)00020-6)). `aep_parallel` is a modified version of `aep` that only updates the posterior covariance matrix Sigma after updating all site parameters (as described in [Cseke and Heskes 2011 JMLR](http://www.jmlr.org/papers/v12/cseke11a.html)). We found that this parallel updating scheme is not only faster but also often leads to better convergence in practice.
<br><br>
`--h2`: The proportion of liability variance explained by the latent variable `g` (i.e. heritability in the context of genetics) (default h2=0.25) 
<br><br>
`--n`: Simulated sample size (default n=500)
<br><br>
`--m`: Number of features used to determine the latent variable *g* (default m=500)
<br><br>
`--frac_cases`: The proportion of case individuals in the sample (default frac_cases=0.5)
<br><br>
`--prev`: The prevalence of cases in the general population (default prev=0.01)
<br><br>
`--kernel`: The kernel matrix. Right now the code natively supports only `linear` and `rbf`. However, the file `kernels.py` contains the building blocks for a large number of kernels that can be used in principle (with code adopted from the [GPML package](http://www.gaussianprocess.org/gpml/code/matlab/doc/)) (defauler kernel=linear)
<br><br>
`--e_dist`: The distribution of the residual variable *e* (i.e. the type of the likelihood function, or the inverse-link function). The code currently supported the options `normal` (probit likelihood) or `logistic` (logistic likelihood). When using logistic likelihood, please normalize the variances by also using the flag `--scale_g` so that the liability variance is 1.0, like in the probit likelihood (default e_dist=probit).





<br><br>
# Contact
For questions and comments, please open a Github issue (preferred) or contact Omer Weissbrod at oweissbrod[at]hsph.harvard.edu





