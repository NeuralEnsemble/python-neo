# -*- coding: utf-8 -*-
"""
This is an example for plotting neo object with maplotlib.
"""

import urllib

import numpy as np
import quantities as pq
from matplotlib import pyplot

import neo

url = 'https://portal.g-node.org/neo/'
distantfile = url + 'neuroexplorer/File_neuroexplorer_2.nex'
localfile = 'File_neuroexplorer_2.nex'
urllib.urlretrieve(distantfile, localfile)


reader = neo.io.NeuroExplorerIO(filename='File_neuroexplorer_2.nex')
bl = reader.read(cascade=True, lazy=False)[0]
for seg in bl.segments:
    fig = pyplot.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.set_title(seg.file_origin)
    mint = 0 * pq.s
    maxt = np.inf * pq.s
    for i, asig in enumerate(seg.analogsignals):
        times = asig.times.rescale('s').magnitude
        asig = asig.rescale('mV').magnitude
        ax1.plot(times, asig)

    trains = [st.rescale('s').magnitude for st in seg.spiketrains]
    colors = pyplot.cm.jet(np.linspace(0, 1, len(seg.spiketrains)))
    ax2.eventplot(trains, colors=colors)

pyplot.show()
