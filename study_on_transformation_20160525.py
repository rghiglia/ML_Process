# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:16:16 2016

@author: rghiglia
"""

# Data transformation
# Understand how rescaling of the data affects PCA
# The thing I want to grasp  better is how can you have high variance ratio
# and low correlation
# For example, keep the correlation the same and vary the variance ratio
# I think what happens is that the beta of the relationship varies
# Actually, isn't the beta kind of the ratio of the variances via OLS estimate?
# Do experiments by varying variance ratio and by varying correlation
# I think if you vary the variance ratio, you'll tilt the principal axis
# Maybe correlation is like changing the ratio of the eigenvalues, so it's
# changing the variance on the projected principal components. It's remakable
# if you can do that independently from changing the variances of the marginals
# well that is true by definition! Will the beta vary?

# 1) Generate correlated Gaussians
# 2) Run PCA (or equivalent principal like in GMM code)
# 3) Look at projection on PCA space

# 4) Same as before but re-scale the data by their marginal standard deviation


from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate a random correlated bivariate dataset
#rs = np.random.RandomState(5)

avg = [1, 1]
sig = [1, 10]
rho = 0.5
n = len(avg)
Rho = (1-rho)*np.eye(n) + rho*np.ones(n)
cov = np.outer(sig,sig)*Rho
nS = 1000
mvn =  multivariate_normal(avg, cov, seed=0)
x1, x2 = mvn.rvs(nS).T
x1 = pd.Series(x1, name="$x_1$")
x2 = pd.Series(x2, name="$x_2$")
sg_hat = [x1.std(), x2.std()]
rho_hat = np.corrcoef(x1, x2)[0,1]


## Show the joint distribution using kernel density estimation
#g = sns.jointplot(x1, x2, kind="kde", size=4, space=0)

# Regression
fit = np.polyfit(x1, x2, 1) # careful: it's phi_n, ..., phi_1, phi_0
fit_fn = np.poly1d(fit)
# fit_fn is now a function which takes in x and returns an estimate for y

# Principal components
lam2, v = np.linalg.eig(cov)
lam = np.sqrt(lam2)



# Better do a standard regression plot
d_lam = 3
ax = plt.subplot(111, aspect='equal')
ax.scatter(x1, x2, alpha=0.1)
ell = Ellipse(xy=avg, width=2*lam[0], height=2*lam[1], 
          angle=np.rad2deg(np.arccos(v[0, 0])), facecolor='none')
ax.add_artist(ell)
ax.plot(x1, fit_fn(x1), '--k', alpha=0.5)
ax.plot(avg[0]+d_lam*np.array([0, v[0,0]]), avg[1]+d_lam*np.array([0, v[1,0]]), 'y', alpha=1) # yeah, this was just an SVD I assume not PCA so may not be ordered
ax.plot(avg[0]-d_lam*np.array([0, v[0,0]]), avg[1]-d_lam*np.array([0, v[1,0]]), 'y', alpha=1)
ax.plot(avg[0]+d_lam*np.array([0, v[0,1]]), avg[1]+d_lam*np.array([0, v[1,1]]), 'r', alpha=1)
ax.plot(avg[0]-d_lam*np.array([0, v[0,1]]), avg[1]-d_lam*np.array([0, v[1,1]]), 'r', alpha=1)
plt.axvline(x1.mean(), color='k', linestyle='--', alpha=0.1)
plt.axhline(x2.mean(), color='k', linestyle='--', alpha=0.1)
plt.title('beta = {:1.2f}'.format(fit[0]))
plt.xlabel(x1.name)
plt.ylabel(x2.name)



# Note: regression line quite different from principal components!

# Qualitatively speaking:
# -) rescaling kind of rotates the axis
# -) the correlation squishes the ellipse from elongated in the 1st PC to a ball
# (isotropic) when rho=0 to elongated along the other dimension that then 
# becomes the new principal component

# I assume beta is a function of both variance ratio and rho, indeed:
# b = rho*sig_y / sig_x
#https://en.wikipedia.org/wiki/Simple_linear_regression
# There are 2 ways beta goes to zero:
# if the variance of x >> variance of y
# if rho goes to zero

# The next big question is:
# Is the projection on PCs variant or invariant under rescaling?
# From the above, I could see it as being invariant.
# However, if invariant why rescale before doing PCA?


X = pd.DataFrame([x1, x2]).T

from sklearn.decomposition import PCA
pca_n2 = PCA(n_components=2)
pca_n2_fit = pca_n2.fit(X)
V = pca_n2_fit.components_
v1 = V[0,:]
v2 = V[1,:]

# Projection of data onto PCs
# pca.transform I think projects the data onto PCAs
X_on_pca_n2 = pca_n2.transform(X)
X_on_pca_n2[:5,:]

pca_n1 = PCA(n_components=1)
pca_n1_fit = pca_n1.fit(X)
V = pca_n1_fit.components_
v1_n1 = V[0,:]

# Projection of data onto PCs
# pca.transform I think projects the data onto PCAs
X_on_pca_n1 = pca_n1.transform(X)
X_on_pca_n1[:5,:]
# Exactly the same as X_on_pca_n2[:5,0]

# Varying the variance ratio the projections also change
# Actually it must change ...
# Imagine the ratio oof variances is very high, then PC1 is almost equal to the
# variable with largest variance


# So the projection changes with both variance ratio and rho
# So I guess if a problem is dominated by a variable that is much more volatile
# actually just much larger
# Wow: never thought about this, actually all variable have the same vol ...
# In the sense that if you rescale the variables, which is like saying
# express in meters instead of feet and then they are all equally volatile!
# So I guess you have to make a decision about that. When you compare two
# variables and you think about is one more important than the other you have
# to ask yourself how does it enter into the problem, i.e. how does it affect
# the output
# For example, suppose you have a portfolio of assets and you cannot rescale
# them, then the behavior will be certainly dominated by scale, i.e. by the
# largest, or more volatile variables
# If instead you use the same assets in a context where you can freely rescale
# them, the individual scale and vol don't matter, it's just about how they
# interact. In that case you want to scale then run PCA

# I would say in ML you always want to rescale, then apply PCA and ADD the PCAs
# as extra features. Why?
# Suppose there is one feature that really drives more of the output because of
# scale, then running a non-scaled PCA will not add anything, so you don't get
# more info. If instead you scale you might get something new.

# Is there an equivalent of PCA for boolean variables?

# Q: suppose I transform the input data by z-scoring, what is the std of the
# data in the transformed space? If that is not 1, which I assume should be
# the case then maybe the 'whitening' option does that
# That would also mean that in the PCA space one variable is more important
# than the other. Duh ... that's the whole point.
