# -*- coding: utf-8 -*-
"""
This is an example demonstrate the lazy load and proxy objects.

"""

import urllib
import neo
import quantities as pq
import numpy as np

url_repo = 'https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/'

# Get Plexon files
distantfile = url_repo + 'micromed/File_micromed_1.TRC'
localfile = './File_micromed_1.TRC'
urllib.request.urlretrieve(distantfile, localfile)

# create a reader
reader = neo.MicromedIO(filename='File_micromed_1.TRC')
reader.parse_header()


lim0, lim1 = -20 * pq.ms, +20 * pq.ms


def apply_my_fancy_average(sig_list):
    """basic average along triggers and then channels
    here we go back to numpy with magnitude
    to be able to use np.stack
    """
    sig_list = [s.magnitude for s in sig_list]
    sigs = np.stack(sig_list, axis=0)
    return np.mean(np.mean(sigs, axis=0), axis=1)


seg = reader.read_segment(lazy=False)
triggers = seg.events[0]
anasig = seg.analogsignals[0]  # here anasig contain the whole recording in memory
all_sig_chunks = []
for t in triggers.times:
    t0, t1 = (t + lim0), (t + lim1)
    anasig_chunk = anasig.time_slice(t0, t1)
    all_sig_chunks.append(anasig_chunk)
m1 = apply_my_fancy_average(all_sig_chunks)


seg = reader.read_segment(lazy=True)
triggers = seg.events[0].load(time_slice=None)  # this load all trigers in memory
anasigproxy = seg.analogsignals[0]  # this is a proxy
all_sig_chunks = []
for t in triggers.times:
    t0, t1 = (t + lim0), (t + lim1)
    anasig_chunk = anasigproxy.load(time_slice=(t0, t1))  # here real data are loaded
    all_sig_chunks.append(anasig_chunk)
m2 = apply_my_fancy_average(all_sig_chunks)

print(m1)
print(m2)
