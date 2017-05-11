#!/usr/bin/python3

import sys
from childes import childes_stats

filename="Brown/Adam/adam01.cha"

if len(sys.argv) > 1:
    filename = sys.argv[1]

stats = childes_stats(filename)

print("{:>22}{:>10}".format('child', 'mother'))
print("utterances: {:>10}{:>10}".format(
    stats['chi_nu'],
    stats['mot_nu']))
print("words:      {:>10}{:>10}".format(
    stats['chi_nwtok'],
    stats['mot_nwtok']))
print("MLU:        {:>10.2f}{:>10.2f}".format(
    stats['chi_nwtok']/stats['chi_nu'],
    stats['mot_nwtok']/stats['mot_nu']))
print("TTR:        {:>10.2f}{:>10.2f}".format(
    stats['chi_nwtyp']/stats['chi_nwtok'],
    stats['mot_nwtyp']/stats['mot_nwtok']))
