#!/usr/bin/env python3

import sys
from childes import childes_utterances

import numpy as np

bos_str = '<'
eos_str = '>'

filename="Brown/Adam/adam55.cha"
if len(sys.argv) > 1:
    filename = sys.argv[1]

def seq_to_ngrams(sequence, ngsize, pad_bos=True, pad_eos=True):
    ngrams = []
    n = ngsize - 1 
    if pad_eos:
        s = sequence + [eos_str]
    else:
        s = sequence
    for i,w in enumerate(s):
        if i < n:
            if pad_bos:
                ngrams.append(tuple([bos_str] * (n - i) + s[0:i+1]))
            else:
                continue
        else:
            ngrams.append(tuple(s[i-n:i+1]))
    return ngrams


#
# The following is a very crude data structure for n-gram counts.
# Since we are interested only in n-grams of a single order 'n',
# we keep both the prefix counts and the counts of the full 'n'-gram
# in a dictionary.
# We also have another hack: the dictionaries also contains a special
# key 'tokens' accumulates the token counts
#
# For most applications you need a better data structure.
#
ngrams = dict()
ngrams['CHI'] = dict()
ngrams['MOT'] = dict()
vocab = set()

for spk, utt in childes_utterances(filename):
    if spk not in {'CHI', 'MOT'}:
        continue
    for word in utt:
        for ng in seq_to_ngrams(list(word.lower()), 2):
            vocab = vocab.union(ng)
            prefix = ng[:-1]
            ngrams[spk][prefix] = 1 + ngrams[spk].get(prefix, 0)
            ngrams[spk][ng] = 1 + ngrams[spk].get(ng, 0)
            ngrams[spk]['tokens'] = 1 + ngrams[spk].get('tokens', 0)

# e2 - we also squeeze the calculation of the data we need for e6 here

vocabs = sorted(vocab)
joint_dist = {'CHI': list(), 'MOT': list()}
for spk in ngrams:
    print(spk)
    for i,c in enumerate(vocabs):
        if i == 0:
            print("{}".format(c), end="")
        else:
            print("\t{}".format(c), end="")
    print()
    for c1 in vocabs:
        print(c1, end="")
        c1_dist = []
        for c2 in vocabs:
            prob = ngrams[spk].get((c1,c2), 0) / ngrams[spk]['tokens']
            print("\t{:.2f}".format(prob), end="")
            c1_dist.append(prob)
        joint_dist[spk].append(c1_dist)
        print()

# e3

def bg_word_prob(word, spk, return_log=False, smooth=False):
    bigrams = seq_to_ngrams(list(word), 2)
    lp = 0
    for c1,c2 in bigrams:
        ng_count = ngrams[spk].get((c1,c2), 0)
        prefix_count = ngrams[spk].get((c1,), 0)
        if smooth:
            cprob = (ng_count + 1) / (prefix_count + len(vocab))
        else:
            if ng_count == 0:
                lp = -float("inf")
                break
            cprob =  ng_count / prefix_count
        lp += np.log2(cprob)

    if return_log:
        return lp
    else:
        return 2**lp

for w in ('mommy', 'yes', 'no', 'good', 'bad', 'my', 'your', 'i', 'you', 'wug', 'nlp'):
    print("{:<10} {:.10f} {:.10f}".format(w, bg_word_prob(w, "CHI"),
                             bg_word_prob(w, "MOT"))
    )
print()

# e4 - was already included in the definition of word_prob()

for w in ('mommy', 'yes', 'no', 'good', 'bad', 'my', 'your', 'i', 'you', 'wug', 'nlp'):
    print("{:<10} {:.10f} {:.10f}".format(w,
        bg_word_prob(w, "CHI", smooth=True),
        bg_word_prob(w, "MOT", smooth=True))
    )
print()

# e5 

best_bg = None
best_prob = 0.0
for c1 in vocab - {'<', '>'}:
    for c2 in vocab - {'<', '>'}:
        bg = ''.join((c1,c2))
        prob = bg_word_prob(bg, 'MOT')
        if prob > best_prob:
            best_prob = prob
            best_bg = bg

print(best_bg, best_prob)

# e6

# pandas makes it easy to index the distributions by letter
import pandas
chi_cond = pandas.DataFrame(joint_dist['CHI'], index=vocabs, columns=vocabs)

for i, row in chi_cond.iterrows():
    chi_cond.loc[i] = row / row.sum()

for wi in range(100):
    w = []
    ch = '<'
    while ch != '>':
        i = np.flatnonzero(np.random.multinomial(1, chi_cond.loc[ch], 1))[0]
        ch = vocabs[i]
        if ch != '>':
            w.append(ch)
    print(''.join(w))
