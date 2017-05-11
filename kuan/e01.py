#!/usr/bin/env python3

# deps:
# main-deps: tabulate, matplotlib


import re
from collections import Counter


def read_chat(file_path):
    """-> iter([(str, str)])
    
    generates pairs (meta: str, content: str) of entries from CHAT (Codes for the
    Human Analysis of Transcripts) file at `file_path`.

    """
    meta, content = "", ""
    with open(file_path) as file:
        for line in file:
            if line.startswith("\t"):
                # this line continues from previous line
                assert meta
                content += line
            else:
                # this line is new
                if meta:
                    # yield previous line if any
                    yield meta, content
                try:
                    # most lines should be meta-tab-content
                    meta, content = line.split("\t")
                except ValueError:
                    # some lines are only a header with no content
                    meta, content = line, ""
        # finally
        if meta:
            yield meta, content


def childes_stats(chat_entries,
                  re_age=re.compile(r"\|CHI\|(\d+);(\d+)\.(\d+)\|"),
                  re_word=re.compile(r"[^\[\]]*\w[^\[\]]*")):
    """-> dict

    returns the following stats from `chat_entries` in a record:

    'age',

    'num_utters_chi', 'num_words_chi', 'num_types_chi', 'mlu_chi', 'ttr_chi',

    'num_utters_mot', 'num_words_mot', 'num_types_mot', 'mlu_mot', 'ttr_mot'.

    mlu: mean length of utterances

    ttr: type token ratio

    """
    # first read the headers to find the age
    for meta, content in chat_entries:
        assert meta.startswith("@")  # make sure to only consume headers
        if "@ID:" == meta and "|CHI|" in content:
            # content = "eng|Brown|CHI|2;3.04|male|typical|MC|Target_Child|||"
            age = re_age.findall(content)
            # age = [('2', '3', '04')]
            y, m, d = map(int, *age)
            age = y + (m / 12.0) + (d / 365.0)
            break
    # then read the utterances
    num_utters_chi, word_freq_chi = 0, Counter()
    num_utters_mot, word_freq_mot = 0, Counter()
    for meta, content in chat_entries:
        if "*CHI:" == meta:
            num_utters_chi += 1
            word_freq_chi.update(w for w in content.split() if re_word.fullmatch(w))
        elif "*MOT:" == meta:
            num_utters_mot += 1
            word_freq_mot.update(w for w in content.split() if re_word.fullmatch(w))
    # return the stats
    num_types_chi, num_words_chi = len(word_freq_chi), sum(word_freq_chi.values())
    num_types_mot, num_words_mot = len(word_freq_mot), sum(word_freq_mot.values())
    return {
        # chi stats
        'age': age,
        'num_utters_chi': num_utters_chi,
        'num_words_chi': num_words_chi,
        'num_types_chi': num_types_chi,
        'mlu_chi': num_words_chi / num_utters_chi,
        'ttr_chi': num_types_chi / num_words_chi,
        # mot stats
        'num_utters_mot': num_utters_mot,
        'num_words_mot': num_words_mot,
        'num_types_mot': num_types_mot,
        'mlu_mot': num_words_mot / num_utters_mot,
        'ttr_mot': num_types_mot / num_words_mot,
    }


def gen_childes_stats(dir_path, extension=".cha"):
    """generates childes stats for all files with `extension` under `dir_path`."""
    if not dir_path.endswith("/"):
        dir_path += "/"
    from glob import glob
    for file_path in glob("{}*{}".format(dir_path, extension)):
        yield childes_stats(read_chat(file_path))


if '__main__' == __name__:
    import sys
    try:
        cha_file, cha_dir = sys.argv[1:]
    except ValueError:
        sys.exit("usage: {} cha_file cha_dir".format(sys.argv[0]))

    print("\n\n* exercise 1\n")
    from pprint import pprint
    pprint(childes_stats(read_chat(cha_file)))

    stats = list(gen_childes_stats(cha_dir))
    age = [x['age'] for x in stats]
    ttr_chi, mlu_chi = [x['ttr_chi'] for x in stats], [x['mlu_chi'] for x in stats]
    ttr_mot, mlu_mot = [x['ttr_mot'] for x in stats], [x['mlu_mot'] for x in stats]

    print("\n\n* exercise 2\n")
    from tabulate import tabulate
    print(tabulate(zip(age, ttr_chi, ttr_mot, mlu_chi, mlu_mot),
                   headers=('age', 'ttr_chi', 'ttr_mot', 'mlu_chi', 'mlu_mot')))

    print("\n\n* exercise 3\n")
    import matplotlib.pyplot as plt
    # upper plot: ttr
    plt.subplot(2, 1, 1)
    plt.plot(age, ttr_chi, age, ttr_mot)
    plt.xlabel("age")
    plt.ylabel("ttr")
    plt.legend(("chi", "mot"))
    # lower plot: mlu
    plt.subplot(2, 1, 2)
    plt.plot(age, mlu_chi, age, mlu_mot)
    plt.xlabel("age")
    plt.ylabel("mlu")
    plt.legend(("chi", "mot"))
    plt.show()
