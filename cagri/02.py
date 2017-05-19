#!/usr/bin/python3

import sys
from glob import glob
from childes import childes_stats
import numpy as np

import numpy as np


# Re-collecting data from the previous exercise

directory="Brown/Adam"

age = []
chimlu = []
motmlu = []
motttr = []
chittr = []

for filename in glob(directory + '/*.cha'):
    stats = childes_stats(filename)
    y, m, d = stats['age']
    age.append(y*365+ m*30+ d)
    chittr.append(stats['chi_nwtyp'] / stats['chi_nwtok'])
    motttr.append(stats['mot_nwtyp'] / stats['mot_nwtok'])
    chimlu.append(stats['chi_nwtok'] / stats['chi_nu'])
    motmlu.append(stats['mot_nwtok'] / stats['mot_nu'])

age = np.array(age)
chimlu = np.array(chimlu)
motmlu = np.array(motmlu)
motttr = np.array(motttr)
chittr = np.array(chittr)


#
# E01 - counting pos tags
#
filename="Brown/Adam/adam01.cha"

stats = childes_stats(filename)

pos_keys = set(stats['chi_pos'].keys())
pos_keys = pos_keys.union(stats['mot_pos'].keys())
pos_keys = sorted(pos_keys)

chipos = np.array([stats['chi_pos'].get(x, 0) for x in pos_keys])
motpos = np.array([stats['mot_pos'].get(x, 0) for x in pos_keys])


# Pandas version for E01
# from pandas import DataFrame
# 
# pos = DataFrame({'mot': stats['mot_pos'], 'chi': stats['chi_pos']})
# pos = pos.replace(np.nan, 0)


# E02: mean, var, sd
# the following should work with lists,
# there are better ways if you work with numpy arrays
motmlu_mean = sum(motmlu)/len(motmlu)
motmlu_var = sum([(x - motmlu_mean)**2 for x in motmlu])/len(motmlu)
motmlu_sd = motmlu_var ** 0.5

chimlu_mean = sum(chimlu)/len(chimlu)
chimlu_var = sum([(x - chimlu_mean)**2 for x in chimlu])/len(chimlu)
chimlu_sd = chimlu_var ** 0.5

# fulll numpy solution:
motmlu.mean()
motmlu.var()
motmlu.std()

chimlu.mean()
chimlu.var()
chimlu.std()

# the rest assumes numpy


# E03
np.corrcoef(age,chimlu)[0,1]
np.corrcoef(chittr,chimlu)[1,0]
np.corrcoef(chimlu,motmlu)[1,0]
np.corrcoef(age,motmlu)[1,0]


# E04

# scipy has function to do it (probably much better)
def cossim(w, v):
    w_norm = np.linalg.norm(w)
    v_norm = np.linalg.norm(v)
    return (np.dot(w, v) / (w_norm * v_norm))

np.dot(chimlu, motmlu)
np.dot(age, motmlu)

cossim(chimlu, motmlu)
cossim(age, motmlu)

# E05
chimlu_c = chimlu - chimlu.mean()

np.dot(chimlu_c, chimlu_c)/chimlu_c.shape[0]

# E06

# scaling:
motpos = motpos / sum(motpos)
chipos = chipos / chipos.sum()

-np.dot(np.log2(chipos), chipos)
-sum([x*y if x != 0 else 0 for x,y in zip(motpos, np.log2(motpos))])
#
# another version for avoiding 0 log 0
# lg_motpos = np.log2(motpos)
# lg_motpos[np.isneginf(lg_motpos)] = np.finfo(np.float).min
