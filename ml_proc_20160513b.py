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


fnmL_trn = dnm + '\\' + fnm_trn
fnmL_tst = dnm + '\\' + fnm_tst

# If data is split into training and testing it might be a good idea to join it
# Actually moving the other way, since it is more likely that you get them separately

# Problem type
# For a supervised problem you need to specify the target label
# If empty it will assume that the problem is unsupervised
nTst = []

if not col_tgt:   # there is a test set, in which case the problem is certainly supervised
    typ_ml = 'un_sup'
else:
    typ_ml = 'sup'
print "The problem is '%s'" % typ_ml.replace('_','').replace('sup', 'supervised')



    
# Load datasets
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


# Classification type
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

print "Observations: \t\t %i\n" % nO
print "Training samples: \t %i" % nTrn
print "Testing samples: \t %i" % nTst
print "---------------------------------------------------"



# Display a description of the dataset
data_trn.info()
data_trn.head()
data_trn.describe()

# I think you need to deal with NaN's at this stage
# You should also think of this as being a processor for both training as well as testing data

# Remove NaN's

# This is a very problem-dependent step
for col, x in data_trn.iteritems():
    # print col
    if x.dtype==np.float64:
        data_trn.ix[x.isnull(), col] = data_trn.ix[x.notnull(), col].mean()
    elif x.dtype==np.int64:
        data_trn.ix[x.isnull(), col] = np.rint(data_trn.ix[x.notnull(), col].mean()).astype(int) # rounding will convert to the closest integer, so ok for categoricals, for integert and continuous it will be the integer closest to the mean, also ok
        


df_types = data_types(data_trn)

data_aug = data_trn.copy()
for nm, sp in df_types['support'].iteritems():
    print
    print nm, sp
    if sp=='continuous':
        print df_types.loc[nm]
        # Create buckets
        nB = 5
        col_num = create_buckets(data_aug[nm], nB)
        data_aug = pd.concat([data_aug, col_num], axis=1)
        out_tmp = cat_series2num(data_aug[col_num.name]) # convert string to numeric level
        col_num = out_tmp['x_num']
        col_num.name = nm + '_num'
        data_aug = pd.concat([data_aug, col_num], axis=1)
    else:
        # Create the couple of (bucket txt value, bucket numerical value)
        # Also create a dictionary of that
        if df_types.ix[nm, 'type'] in (np.int64, np.float64):
            col_txt = nm + '_' + data_aug[nm].astype(str) # convert numeric level to string
            col_txt.name = nm + '_lev'
            data_aug = pd.concat([data_aug, col_txt], axis=1)
        elif df_types.ix[nm, 'type']==np.object:
            out_tmp = cat_series2num(data_aug[nm]) # convert string to numeric level
            col_num = out_tmp['x_num']
            col_num.name = nm + '_num'
            data_aug = pd.concat([data_aug, col_num], axis=1)

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



