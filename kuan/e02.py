#!/usr/bin/env python3

# deps: numpy
# main-deps: e01.py, tabulate


import re
from collections import Counter
import numpy as np
import numpy.ma as ma


def childes_pos(chat_entries, re_pos=re.compile(r"([^\s\|:]+)(?:\||:)[^\s]+")):
    """-> dict

    returns the following data from `chat_entries` in a record:

    'postags': [str], 'pos_freq_chi': [int], 'pos_freq_mot: [int].

    """
    pos_freq_chi = Counter()
    pos_freq_mot = Counter()
    pos_freq = pos_freq_chi, pos_freq_mot
    state = 2  # 0: chi, 1: mot, 2: ignore
    for meta, content in chat_entries:
        if meta.startswith("*"):
            if "*CHI:" == meta:
                state = 0
            elif "*MOT:" == meta:
                state = 1
            else:
                state = 2
        elif meta.startswith("%"):
            if "%mor:" == meta and 2 > state:
                pos_freq[state].update(t for t in re_pos.findall(content))
        else:
            state = 2
    postags = sorted(set.union(*map(set, pos_freq)))
    return {
        'postags': postags,
        'pos_freq_chi': [pos_freq_chi[pos] for pos in postags],
        'pos_freq_mot': [pos_freq_mot[pos] for pos in postags],
    }


def mean_var_sd(x):
    """np.ndarray -> dict

    assert 2 <= n.size

    returns the following stats about `x` in a record:

    'mean', 'var', 'sd'.

    """
    n = x.size
    assert 2 <= n
    mean = x.sum() / n
    diff = x - mean
    var = np.vdot(diff, diff) / (n - 1)
    sd = var ** 0.5
    return {
        'mean': mean,
        'var': var,
        'sd': sd,
    }


def cos_sim(u, v):
    """np.ndarray, np.ndarray -> float

    returns the cosine similarity between `u` and `v`.

    """
    return np.vdot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def center(x):
    """np.ndarray -> np.ndarray

    centers `x` around the origin. not in place.

    """
    return x - x.mean()


def scale(x, p=2, inplace=False):
    """np.ndarray -> np.ndarray

    scales `x` by its `p`-norm. not in place.

    """
    return x / np.linalg.norm(x, ord=p)


def cross_entropy(p, q, base=2):
    """np.ndarray, np.ndarray -> float"""
    q = ma.array(q, mask=(q == 0))
    return - np.vdot(p, ma.log(q)) / np.log(base)


def entropy(x, base=2):
    """np.ndarray -> float"""
    return cross_entropy(x, x, base)


if '__main__' == __name__:
    import sys
    try:
        cha_file, cha_dir = sys.argv[1:]
    except ValueError:
        sys.exit("usage: {} cha_file cha_dir".format(sys.argv[0]))

    from e01 import read_chat, childes_stats, gen_childes_stats

    print("\n\n* exercise 1\n")
    from tabulate import tabulate
    pos = childes_pos(read_chat(cha_file))
    print(tabulate(zip(pos['postags'], pos['pos_freq_chi'], pos['pos_freq_mot']),
                   headers=("postags", "chipos", "motpos")))

    stats = list(gen_childes_stats(cha_dir))
    age = np.fromiter((x['age'] for x in stats), np.float32)
    ttr_chi = np.fromiter((x['ttr_chi'] for x in stats), np.float32)
    mlu_chi = np.fromiter((x['mlu_chi'] for x in stats), np.float32)
    ttr_mot = np.fromiter((x['ttr_mot'] for x in stats), np.float32)
    mlu_mot = np.fromiter((x['mlu_mot'] for x in stats), np.float32)

    print("\n\n* exercise 2\n")
    cols = 'name', 'mean', 'var', 'sd'
    stats_mlu_chi = mean_var_sd(mlu_chi)
    stats_mlu_chi['name'] = "chimlu"
    stats_mlu_chi_np = {
        'name': "chimlu_np",
        'mean': np.mean(mlu_chi),
        'var': np.var(mlu_chi),
        'sd': np.std(mlu_chi),
    }
    stats_mlu_mot = mean_var_sd(mlu_mot)
    stats_mlu_mot['name'] = "motmlu"
    stats_mlu_mot_np = {
        'name': "motmlu_np",
        'mean': np.mean(mlu_mot),
        'var': np.var(mlu_mot),
        'sd': np.std(mlu_mot),
    }
    print(tabulate([[stats[col] for col in cols] for stats in
                    (stats_mlu_chi, stats_mlu_chi_np, stats_mlu_mot, stats_mlu_mot_np)],
                   headers=cols))

    print("\n\n* exercise 3\n")
    print(tabulate((
        ("age",    "chimlu", np.corrcoef(age,     mlu_chi)[0, 1]),
        ("chittr", "chimlu", np.corrcoef(ttr_chi, mlu_chi)[0, 1]),
        ("chimlu", "motmlu", np.corrcoef(mlu_chi, mlu_mot)[0, 1]),
        ("age",    "motmlu", np.corrcoef(age,     mlu_mot)[0, 1]),
    ), headers=("x", "y", "corrcoef")))

    print("\n\n* exercise 4\n")
    print(tabulate((
        ("chimlu", "motmlu", mlu_chi @ mlu_mot, cos_sim(mlu_chi, mlu_mot)),
        ("age",    "motmlu", age @ mlu_mot,     cos_sim(age, mlu_mot)),
    ), headers=("x", "y", "dot", "sim")))

    print("\n\n* exercise 5\n")
    mlu_chi_c = center(mlu_chi)
    print((mlu_chi_c @ mlu_chi_c) / mlu_chi_c.size)

    print("\n\n* exercise 6\n")
    pos_freq_chi = scale(pos['pos_freq_chi'], p=1)
    pos_freq_mot = scale(pos['pos_freq_mot'], p=1)
    print(tabulate((
        ("chipos", "motpos", entropy(pos_freq_chi), cross_entropy(pos_freq_chi, pos_freq_mot)),
        ("motpos", "chipos", entropy(pos_freq_mot), cross_entropy(pos_freq_mot, pos_freq_chi)),
    ), headers=("p", "q", "ent_p", "p_xent_q")))
