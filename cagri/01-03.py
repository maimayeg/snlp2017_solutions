#!/usr/bin/python3

import sys
from glob import glob
from childes import childes_stats

directory="Brown/Adam"

if len(sys.argv) > 1:
    directory = sys.argv[1]


chi_age = []
chi_mlu = []
mot_mlu = []
mot_ttr = []
chi_ttr = []

for filename in glob(directory + '/*.cha'):
    stats = childes_stats(filename)
    y, m, d = stats['age']
    chi_age.append(y*365+ m*30+ d)
    chi_ttr.append(stats['chi_nwtyp'] / stats['chi_nwtok'])
    mot_ttr.append(stats['mot_nwtyp'] / stats['mot_nwtok'])
    chi_mlu.append(stats['chi_nwtok'] / stats['chi_nu'])
    mot_mlu.append(stats['mot_nwtok'] / stats['mot_nu'])


import matplotlib.pyplot as plt

plt.plot(chi_age, chi_ttr, chi_age, mot_ttr)
plt.xlabel("Child's age (days)")
plt.ylabel('TTR')
plt.legend(['chi_ttr', 'mot_ttr'])
plt.savefig('ttr.pdf')
plt.close()

plt.plot(chi_age, chi_mlu, chi_age, mot_mlu)
plt.xlabel("Child's age (days)")
plt.ylabel('MLU (words)')
plt.legend(['chi_mlu', 'mot_mlu'])
plt.savefig('mlu.pdf')
