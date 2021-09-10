# -*- coding: utf-8 -*-

"""
This generate diagram of the (raw)ios and formats


Author: Julia Sprenger
"""

import pygraphviz
import neo
# from datetime import datetime
#
# import numpy as np
# import quantities as pq
# from matplotlib import pyplot
# from matplotlib.patches import Rectangle, ArrowStyle, FancyArrowPatch
# from matplotlib.font_manager import FontProperties
#
# from neo.test.generate_datasets import fake_neo
#
# line_heigth = .22
# fontsize = 10.5
# left_text_shift = .1
# dpi = 100


default_style = {'shape': 'rectangle',
                 'color': 'black',
                 'fontcolor': 'black'}
IO_style = default_style.copy()
IO_style['fontsize'] = '30'
IO_style['penwidth'] = 6

styles = {'IO': {'ro': IO_style.copy(),
                 'rw': IO_style.copy(),
                 'raw': IO_style.copy()
                 },
          'main': default_style.copy(),
          'ext': default_style.copy()}

styles['IO']['ro']['color'] = '#20B2AA '
styles['IO']['rw']['color'] = '#4169E1 '
styles['IO']['raw']['color'] = '#008080 '
styles['ext']['shape'] = 'circle'
styles['ext']['fillcolor'] = 'red'
styles['ext']['style'] = 'filled'
# styles['ext']['fixedsize'] = 'True'


def generate_diagram(filename, plot_extensions=False):
    dia = pygraphviz.AGraph(strict=False, splines='true')
    G=dia
    G.node_attr['fontname'] = 'Arial'
    # G.node_attr['shape'] = 'circle'
    # G.node_attr['fixedsize'] = 'true'
    # G.node_attr['sep'] = '-100'
    G.node_attr['fixedsize'] = 'False'
    # G.graph_attr['overlap'] = 'False'
    G.graph_attr['packMode'] = 'clust'
    # G.graph_attr['levelsgap'] = -500
    G.node_attr['fontsize'] = '20'
    G.edge_attr['minlen'] = '0'
    # G.node_attr['style'] = 'filled'
    # G.graph_attr['outputorder'] = 'edgesfirst'
    # G.graph_attr['splines'] = "compound"
    G.graph_attr['label'] = "NEO {}".format(neo.__version__)
    G.graph_attr['ratio'] = '1.0'
    # G.edge_attr['color'] = '#1100FF'


    G.edge_attr['style'] = 'setlinewidth(4)'

    dia.add_node('NEO', shape='circle', fontsize=50)

    for io in neo.io.iolist:
        io_name = io.name
        rawio = False
        if issubclass(io, neo.io.basefromrawio.BaseFromRaw):
            rawio = True
            if io_name == 'BaseIO':
                io_name = io.__name__.rstrip('RawIO')
        if io_name is None:
            try:
                io_name = io.__name__.rstrip('IO')
            except:
                continue
        if 'example' in io_name:
            continue

        if io.is_writable and io.is_readable:
            mode = 'rw'
        elif io.is_readable:
            mode = 'ro'
        if rawio:
            mode = 'raw'

        dia.add_node(io_name, **styles['IO'][mode])
        dia.add_edge('NEO', io_name)

        if plot_extensions:
            for ext in io.extensions:
                dia.add_node(ext, **styles['ext'])
                dia.add_edge(io_name, ext, minlen=0)

    dia.layout(prog='fdp') #neato, dot, twopi, circo, fdp, nop, wc, acyclic, gvpr, gvcolor,
    # ccomps, sccmap, tred, sfdp.
    for ext in ['png', 'svg', 'eps']:
        dia.draw('{}.{}'.format(filename, ext))



if __name__ == '__main__':
    generate_diagram('IODiagram', plot_extensions=False)
    generate_diagram('IODiagram_ext', plot_extensions=True)
    # pyplot.show()
