import re

re_remove = re.compile(r'\[[^\]]*\]|\([^)]*\)|[?.,!]')
age_re = re.compile(r'@ID:.*\|CHI\|(?P<age>[0-9;.]+).*')

def childes_stats(filename):
    mot_w = set()
    chi_w = set()

    stats = {'mot_nwtok' : 0,
             'chi_nwtok' : 0,
             'mot_nwtyp' : 0,
             'chi_nwtyp' : 0,
             'mot_nu' : 0,
             'chi_nu' : 0,
             'age' : None,
             'chi_pos' : dict(),
             'mot_pos' : dict(),
    }
    lines = open(filename, "r").readlines()

    for i in range(len(lines)):
        line = lines[i]
        j = i + 1
        while j < len(lines) and lines[j].startswith('\t'):
            line += ' ' + lines[j].strip()
            j += 1
        if line.startswith("*"): # main line 
            speaker, utterance = line.strip().split('\t', 1)
            utterance = re_remove.sub('', utterance)
            words = utterance.split()
            if speaker == '*CHI:':
                chi_w = chi_w.union(words)
                stats['chi_nwtok'] += len(words)
                stats['chi_nu'] += 1
            elif speaker == '*MOT:':
                mot_w = mot_w.union(words)
                stats['mot_nwtok'] += len(words)
                stats['mot_nu'] += 1
        elif line.startswith('@'): # header
            m = age_re.match(line)
            if m:
                stats['age'] = [int(x) for x in re.split( r'[;.]', m.group(1))]
        elif line.startswith('%mor:'): # morphology line
            tokens = line[5:].split()
            for token in tokens:
                if '|' in token:
                    pos = token.split('|', 1)[0].split(':', 1)[0]
                    if speaker == '*MOT:':
                        stats['mot_pos'][pos] = 1+stats['mot_pos'].get(pos, 0)
                    elif speaker == '*CHI:':
                        stats['chi_pos'][pos] = 1+stats['chi_pos'].get(pos, 0)

    stats['chi_nwtyp'] = len(chi_w)
    stats['mot_nwtyp'] = len(mot_w)
    
    return stats
