#!/usr/bin/python3


from glob import glob
import sys
from childes import childes_stats

directory="Brown/Adam"
if len(sys.argv) > 1:
    directory = sys.argv[1]


print("chi_age\tchi_ttr\tmot_ttr\tchi_mlu\tmot_mlu")
for filename in glob(directory + '/*.cha'):
    stats = childes_stats(filename)
    print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
            stats['age'],
            stats['chi_nwtyp'] / stats['chi_nwtok'],
            stats['mot_nwtyp'] / stats['mot_nwtok'],
            stats['chi_nwtok'] / stats['chi_nu'],
            stats['mot_nwtok'] / stats['mot_nu']
            )
    )
