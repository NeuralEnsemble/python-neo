"""
This is an example for plotting neo object with maplotlib.
"""

import neo
import urllib
from matplotlib import pyplot
import numpy

distantfile = 'https://portal.g-node.org/neo/neuroexplorer/File_neuroexplorer_2.nex'
localfile = 'File_neuroexplorer_2.nex'
urllib.urlretrieve(distantfile, localfile)


reader = neo.io.NeuroExplorerIO(filename = 'File_neuroexplorer_2.nex')
bl = reader.read(cascade = True, lazy = False)
for seg in bl.segments:
    fig = pyplot.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex = ax1)
    ax1.set_title(seg.file_origin)
    for asig in seg.analogsignals:
        ax1.plot(asig.times, asig)
    for s,st in enumerate(seg.spiketrains):
        print st.units
        ax2.plot(st, s*numpy.ones(st.size), linestyle = 'None', 
                    marker = '|', color = 'k')
pyplot.show()
