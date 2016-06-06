# -*- coding: utf-8 -*-
"""
Created on Fri May 13 08:23:53 2016

@author: rghiglia
"""

# Starting to build a process
# There are many options, types of problems, etc.
# I need to start somewhere and refine as we go

# -----------------------------------------------------------------------------
# Path, Packages, Etc.
# -----------------------------------------------------------------------------
import sys; sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND')
from rg_toolbox_data import data_types, cat_series2num, create_buckets
#from rg_toolbox_data import cut_to_df, preproc_data, cat2num

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from matplotlib import pyplot as pl
from IPython.display import display
import re
from sklearn.cross_validation import StratifiedShuffleSplit


# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------
dnm = 'C:\Users\rghiglia\Documents\ML_ND'
fnm_trn = ''
fnm_tst = ''
col_tgt = ''


# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------

# Specify files

## Example #1: Supervised, Commingled Data
#dnm = r'C:\Users\rghiglia\Documents\ML_ND\student_intervention'
#fnm_trn = 'student-data.csv'
#fnm_tst = ''
#col_tgt = 'passed'

# Example #2: Supervised, Separate Data
dnm = r'C:\Users\rghiglia\Documents\ML_ND\Kaggle\Titanic'
fnm_trn = 'train.csv'
fnm_tst = 'test.csv'
col_tgt = 'Survived'

## Example #3
#dnm = r'C:\Users\rghiglia\Documents\ML_ND\customer_segment'
#fnm_trn = 'customers.csv'

# Data files
fnmL_trn = dnm + '\\' + fnm_trn
fnmL_tst = dnm + '\\' + fnm_tst

print
print "Training data: '%s'" % fnmL_trn
print "Test data    : '%s'" % fnmL_tst
print


# If data is split into training and testing it might be a good idea to join it
# Actually moving the other way, since it is more likely that you get them separately




# Type of problem
# For a supervised problem you need to specify the target label
# If empty it will assume that the problem is unsupervised
nTst = []

if not col_tgt:   # there is a test set, in which case the problem is certainly supervised
    typ_ml = 'un_sup'
else:
    typ_ml = 'sup'
print "\nThe problem is '%s'" % typ_ml.replace('_','').replace('sup', 'supervised')



    
# Load datasets
print
try:
    data_tmp = pd.read_csv(fnmL_trn)
    print "Training dataset has {} samples with {} features.".format(*data_tmp.shape)
    if typ_ml=='sup':
        if not fnm_tst: # there is no separate training and test data are commingled
            
            # Shuffle and split training and test
            if not nTst: nTst = np.min([100, int(data_tmp.shape[0]*0.15)])
            print "Test samples = %i" % nTst
            sss0 = StratifiedShuffleSplit(data_tmp[col_tgt], 1, test_size=nTst, random_state=0)
            for ix_trn, ix_tst in sss0: _ = False
            data_tmp_trn = data_tmp.ix[ix_trn] # still has all colmuns (including output), but rows represent only training data
            data_tmp_tst = data_tmp.ix[ix_tst] # still has all colmuns (including output), but rows represent only testing data
            
            # Extract sets
            y_trn = data_tmp_trn[col_tgt]
            y_tst = data_tmp_tst[col_tgt]
            data_trn = data_tmp_trn.drop(col_tgt, axis=1)
            data_tst = data_tmp_trn[col_tgt]
            
        else: # there is a separate file
            y_trn = data_tmp[col_tgt]
            data_trn = data_tmp.drop(col_tgt, axis=1)
            try:
                data_tst = pd.read_csv(fnmL_tst)
                y_tst = []
                print "Testing dataset has {} samples with {} features.".format(*data_tst.shape)
            except:
                print "Dataset could not be loaded. Is the dataset missing?"
                
except:
    print "Dataset could not be loaded. Is the dataset missing?"


# Type of output
nTrn, nF = data_trn.shape
nTst = data_tst.shape[0]
nO = nTrn + nTst

if type(y_trn[0])==np.float64:
    typ_cls = 'Continuous'
else:
    if type(y_trn[0])==np.int64:
        print "Output is integer"
    else:
        print "Output is text, it will need to change to numerical value"
    yU = sorted(set(y_trn))
    if len(yU)==2:
        typ_cls = 'Binary'
    elif len(yU)>2:
        typ_cls = 'Multi-class'
    else:
        typ_cls = '"Something is wrong"'


# Print stats
print "\n"
print "---------------------------------------------------"
print "Data"
print "---------------------------------------------------"
print "Features: \t\t %i" % nF
print "Target column: \t\t '%s'" % col_tgt
print "Output type:  \t\t '%s'\n" % typ_cls
if len(yU)==2:
    ix0 = y_trn[y_trn==yU[0]].index
    ix1 = y_trn[y_trn==yU[1]].index
    print "%s = %i \t\t %1.1f%%" % (col_tgt, yU[0], float(len(ix0))/nTrn*100)
    print "%s = %i \t\t %1.1f%%" % (col_tgt, yU[1], float(len(ix1))/nTrn*100)

print "Total observations: \t %i\n" % nO
print "Training samples: \t %i" % nTrn
print "Testing samples: \t %i" % nTst
print "---------------------------------------------------"


# Display a description of the dataset
print "\n Data Info"
display(data_trn.info())
print "\n Data Head"
display(data_trn.head())
print "\n Data Description"
display(pd.DataFrame(np.round(data_trn.describe(),1)))


# -----------------------------------------------------------------------------
# Remove Unwanted Columns or Features
# -----------------------------------------------------------------------------
col_drop = ['PassengerId']
data_trn.drop(col_drop, inplace=True, axis=1)
data_tst.drop(col_drop, inplace=True, axis=1)


# -----------------------------------------------------------------------------
# Data Exploration (before or after NaN treatment?)
# -----------------------------------------------------------------------------

# Let's try before
# Maybe works for simple stats, but might be a problem if you apply DT's for example

# First look at marginals
# Actually for supervised learning problems you should leverage the stuff from titanic proj from MLND

sys.path.append(r'C:\Users\rghiglia\Documents\ML_ND\titanic_survival_exploration')
from titanic_visualizations import survival_stats
survival_stats(data_trn, y_trn, 'Sex')
# Ok, now try to extract the viz code and make it more generic at least with single condition

# Grouping
feat = 'Sex'
df_grp = data_trn.groupby(feat)
display(np.round(df_grp.describe(),1))

# But what you really want to know is just sex and survivorship
df_tmp = pd.concat([data_trn[feat], y_trn], axis=1)
df_grp = df_tmp.groupby([df_tmp[col_tgt], df_tmp[feat]]).count()
display(df_grp)

df_tmp = pd.concat([data_trn, y_trn], axis=1)
df_grp = df_tmp.groupby([feat, col_tgt]).count()
df_grp = df_grp.ix[:,0].unstack()
df_grp['Tot'] = df_grp.sum(axis=1)
df_grp.ix['Tot'] = df_grp.sum(axis=0)
display(df_grp)

# Check
n_fem = df_tmp[feat][df_tmp[feat]=='female'].count()
n_mal = df_tmp[feat][df_tmp[feat]=='male'].count()
n_fem + n_mal == nTrn
n_femS = df_tmp[feat][(df_tmp[feat]=='female') & (df_tmp[col_tgt]==1)].count()
n_femN = df_tmp[feat][(df_tmp[feat]=='female') & (df_tmp[col_tgt]==0)].count()
n_malS = df_tmp[feat][(df_tmp[feat]=='male') & (df_tmp[col_tgt]==1)].count()
n_malN = df_tmp[feat][(df_tmp[feat]=='male') & (df_tmp[col_tgt]==0)].count()
n_femS + n_femN == n_fem
n_malS + n_malN == n_mal

print
df_grp_pct = summarize_feat_supervised(data_trn, y_trn, feat, col_tgt, isPct=1)
display(df_grp_pct)

print
df_grp = summarize_feat_supervised(data_trn, y_trn, feat, col_tgt, isPct=0)
display(df_grp)

# Add F1 score
y_feat = y_trn.copy()
y_feat[data_trn[feat]=='male'] = 0
y_feat[data_trn[feat]=='female'] = 1

from sklearn.metrics import f1_score
f1_feat = f1_score(y_trn,y_feat)


## Visualize
#import seaborn as sns
#pl.subplots(figsize=(6, 4))
#h = sns.heatmap(df_grp_pct.transpose(), annot=True, cmap='YlGnBu');
#pl.subplots_adjust(top=0.9)
#h.figure.suptitle(col_tgt + ' by ' + feat  + ' F1 = {0:1.2f}'.format(f1_feat));
heat_feat(df_grp_pct, col_tgt, feat, f1_feat)


feat = 'Pclass'
print
df_grp = summarize_feat_supervised(data_trn, y_trn, feat, col_tgt, isPct=0)
display(df_grp)

print
df_grp_pct = summarize_feat_supervised(data_trn, y_trn, feat, col_tgt, isPct=1)
display(df_grp_pct)


# Add F1 score
y_feat = y_trn.copy()
for i in set(data_trn[feat]):
    y_feat[data_trn[feat]==i] = df_grp_pct.columns[(df_grp_pct.loc[i]).argmax()]

heat_feat(df_grp_pct, col_tgt, feat, data_trn, y_trn)


import matplotlib.pyplot as plt
# Set the width of each bar
wd = 0.4

# Display each category's survival rates
frame = df_grp
for i in np.arange(len(frame)):
    nonsurv_bar = plt.bar(i-wd, nTrn-frame.ix[i], width=wd, color='r')
    surv_bar = plt.bar(i, frame.ix[i], width=wd, color='g')

    plt.xticks(np.arange(len(frame)), values)
    plt.legend((nonsurv_bar[0], surv_bar[0]),('Did not survive', 'Survived'), framealpha = 0.8)

# Common attributes for plot formatting
plt.xlabel(key)
plt.ylabel('Number of Passengers')
plt.title('Passenger Survival Statistics With \'%s\' Feature'%(key))
plt.show()



# Then look at correlations
# In principle I'd like to see some joint stats
pd.scatter_matrix(data_trn, alpha=0.3, figsize=(5,6), diagonal='kde');
# In case of mixed data this really doesn't give you a good sense of relationships
# I guess you might split into continuous and categorical, but still how about the relationship between continuous and categorical?
# Note: L-shaped pairs of variables: if you sum or take the product you get stuff that is more constant or maybe linear, maybe it tells you something
# You have all kind of 'garbage' continuous with categorical or binary and 
# all combos of those

# Maybe you can try to see a pair and the class
clr = ['r', 'b', 'y', 'm', 'c', 'k']
col_i = 'SibSp'
col_j = 'Parch'
# Adding some random noise to distinguish the dots
Z = DataFrame(np.random.rand(nTrn,2), index=data_trn.index)
dxy = 0.45
for j in range(len(set(y_trn))):
    ix = y_trn==j
    pl.scatter(data_trn.ix[ix, col_i]+dxy*Z.ix[ix, 0], data_trn.ix[ix, col_j]+dxy*Z.ix[ix, 1], color=clr[j], alpha=0.1)
# Finally!

col_i = 'Age'
col_j = 'Fare'
dxy = 0
for j in range(len(set(y_trn))):
    ix = y_trn==j
    pl.scatter(data_trn.ix[ix, col_i]+dxy*Z.ix[ix, 0], np.log(data_trn.ix[ix, col_j]+dxy*Z.ix[ix, 1]), color=clr[j], alpha=0.1)
pl.xlabel(col_i)
pl.ylabel('log ' + col_j)
# Finally!

# Try PCA anyways?

# Where do you decide about scaling?

# Can you do a model conditional on certain variables?
# Say you decide that one sex is a good determinant. Then you allpy a ML model
# only to the male population



# Select some representative values? Probably a good practice





# -----------------------------------------------------------------------------
# NaN Treatment
# -----------------------------------------------------------------------------

# I think you need to deal with NaN's at this stage
# Important: You should also think of this as being a processor for both training as well as testing data

# This is a very problem-dependent step
from rg_toolbox_ml_nan import nan2avg
data_trn = nan2avg(data_trn)
data_tst = nan2avg(data_trn, data_tst)





# -----------------------------------------------------------------------------
# Data Augmentation
# -----------------------------------------------------------------------------

# This is also a very problem-specific algo
# And this also needs to work as a method both on training as well as on test data
from rg_toolbox_ml import data_augment

# THIS DOESN'T WORK YET ... REVIEW
data_trn_aug = data_augment(data_trn)
data_tst_aug = data_augment(data_tst)

# Feature data types
# Probably the best way to think about it is a separation in 2 dimensions:
# Suport: continuous vs. discrete (categorical)
# Type: numeric, text, time stamp
# 1) Assign (support, data_type)
# 2) Screen for support first





# If float: bucket (assign tag 'continuous')
# If integer and # of uniques exceeds a threshold: bucket (tag 'continuous'), otherwise create text labels (tag 'categorical')
# If text and # of uniques below a threshold: produce numerical representation
# Else don't do anything but suggest custom treatment

## Each feature should have 2 properties:
#typ_support = \in {'continuous', 'categorical'}
#typ_type = \in {'numeric', 'text'}
# If continunuous certainly numeric, but if categorical it could be either
# If text certainly categorical, but if numeric it could be either

# Actually you could describe continuous and text, e.g. full names, descriptions, etc.

# Ok, restart from here ...


# At the end of this:
# Each continuous/numeric feature has 3 columns: original (continuous/numeric), bucketed (categorical/text) bucketed (categorical/numeric)
# Each categorical/numeric feature has 2 columns: original (categorical/numeric), text-ed version (categorical/text)
# Categorical/text features could have 1 columns: original (categorical/numeric), text-ed version (categorical/text)



