# -*- coding: utf-8 -*-

"""
This generate diagram in .png and .svg from neo.core


Author: sgarcia
"""

from datetime import datetime

import numpy as np
import quantities as pq
from matplotlib import pyplot
from matplotlib.patches import Rectangle, ArrowStyle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

from neo.test.generate_datasets import fake_neo

line_heigth = .22
fontsize = 10.5
left_text_shift = .1
dpi = 100


def get_rect_height(name, obj):
    '''
    calculate rectangle height
    '''
    nlines = 1.5
    nlines += len(getattr(obj, '_all_attrs', []))
    nlines += len(getattr(obj, '_single_child_objects', []))
    nlines += len(getattr(obj, '_multi_child_objects', []))
    nlines += len(getattr(obj, '_multi_parent_objects', []))
    return nlines*line_heigth


def annotate(ax, coord1, coord2, connectionstyle, color, alpha):
    arrowprops = dict(arrowstyle='fancy',
                      #~ patchB=p,
                      shrinkA=.3, shrinkB=.3,
                      fc=color, ec=color,
                      connectionstyle=connectionstyle,
                      alpha=alpha)
    bbox = dict(boxstyle="square", fc="w")
    a = ax.annotate('', coord1, coord2,
                    #xycoords="figure fraction",
                    #textcoords="figure fraction",
                    ha="right", va="center",
                    size=fontsize,
                    arrowprops=arrowprops,
                    bbox=bbox)
    a.set_zorder(-4)


def calc_coordinates(pos, height):
    x = pos[0]
    y = pos[1] + height - line_heigth*.5

    return pos[0], y


def generate_diagram(filename, rect_pos, rect_width, figsize):
    rw = rect_width

    fig = pyplot.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    all_h = {}
    objs = {}
    for name in rect_pos:
        objs[name] = fake_neo(name)
        all_h[name] = get_rect_height(name, objs[name])

    # draw connections
    color = ['c', 'm', 'y']
    alpha = [1., 1., 0.3]
    for name, pos in rect_pos.items():
        obj = objs[name]
        relationships = [getattr(obj, '_single_child_objects', []),
                         getattr(obj, '_multi_child_objects', []),
                         getattr(obj, '_child_properties', [])]

        for r in range(3):
            for ch_name in relationships[r]:
                x1, y1 = calc_coordinates(rect_pos[ch_name], all_h[ch_name])
                x2, y2 = calc_coordinates(pos, all_h[name])

                if r in [0, 2]:
                    x2 += rect_width
                    connectionstyle = "arc3,rad=-0.2"
                elif y2 >= y1:
                    connectionstyle = "arc3,rad=0.7"
                else:
                    connectionstyle = "arc3,rad=-0.7"

                annotate(ax=ax, coord1=(x1, y1), coord2=(x2, y2),
                         connectionstyle=connectionstyle,
                         color=color[r], alpha=alpha[r])

    # draw boxes
    for name, pos in rect_pos.items():
        htotal = all_h[name]
        obj = objs[name]
        allrelationship = (getattr(obj, '_child_containers', []) +
                           getattr(obj, '_multi_parent_containers', []))

        rect = Rectangle(pos, rect_width, htotal,
                         facecolor='w', edgecolor='k', linewidth=2.)
        ax.add_patch(rect)

        # title green
        pos2 = pos[0], pos[1]+htotal - line_heigth*1.5
        rect = Rectangle(pos2, rect_width, line_heigth*1.5,
                         facecolor='g', edgecolor='k', alpha=.5, linewidth=2.)
        ax.add_patch(rect)

        # single relationship
        relationship = getattr(obj, '_single_child_objects', [])
        pos2 = pos[1] + htotal - line_heigth*(1.5+len(relationship))
        rect_height = len(relationship)*line_heigth

        rect = Rectangle((pos[0], pos2), rect_width, rect_height,
                         facecolor='c', edgecolor='k', alpha=.5)
        ax.add_patch(rect)

        # multi relationship
        relationship = (getattr(obj, '_multi_child_objects', []) +
                        getattr(obj, '_multi_parent_containers', []))
        pos2 = (pos[1]+htotal - line_heigth*(1.5+len(relationship)) -
                rect_height)
        rect_height = len(relationship)*line_heigth

        rect = Rectangle((pos[0], pos2), rect_width, rect_height,
                         facecolor='m', edgecolor='k', alpha=.5)
        ax.add_patch(rect)

        # necessary attr
        pos2 = (pos[1]+htotal -
                line_heigth*(1.5+len(allrelationship) +
                             len(obj._necessary_attrs)))
        rect = Rectangle((pos[0], pos2), rect_width,
                         line_heigth*len(obj._necessary_attrs),
                         facecolor='r', edgecolor='k', alpha=.5)
        ax.add_patch(rect)

        # name
        if hasattr(obj, '_quantity_attr'):
            post = '* '
        else:
            post = ''
        ax.text(pos[0]+rect_width/2., pos[1]+htotal - line_heigth*1.5/2.,
                name+post,
                horizontalalignment='center', verticalalignment='center',
                fontsize=fontsize+2,
                fontproperties=FontProperties(weight='bold'),
                )

        #relationship
        for i, relat in enumerate(allrelationship):
            ax.text(pos[0]+left_text_shift, pos[1]+htotal - line_heigth*(i+2),
                    relat+': list',
                    horizontalalignment='left', verticalalignment='center',
                    fontsize=fontsize,
                    )
        # attributes
        for i, attr in enumerate(obj._all_attrs):
            attrname, attrtype = attr[0], attr[1]
            t1 = attrname
            if (hasattr(obj, '_quantity_attr') and
                    obj._quantity_attr == attrname):
                t1 = attrname+'(object itself)'
            else:
                t1 = attrname

            if attrtype == pq.Quantity:
                if attr[2] == 0:
                    t2 = 'Quantity scalar'
                else:
                    t2 = 'Quantity %dD' % attr[2]
            elif attrtype == np.ndarray:
                t2 = "np.ndarray %dD dt='%s'" % (attr[2], attr[3].kind)
            elif attrtype == datetime:
                t2 = 'datetime'
            else:
                t2 = attrtype.__name__

            t = t1+' :  '+t2
            ax.text(pos[0]+left_text_shift,
                    pos[1]+htotal - line_heigth*(i+len(allrelationship)+2),
                    t,
                    horizontalalignment='left', verticalalignment='center',
                    fontsize=fontsize,
                    )

    xlim, ylim = figsize
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename, dpi=dpi)


def generate_diagram_simple():
    figsize = (18, 12)
    rw = rect_width = 3.
    bf = blank_fact = 1.2
    rect_pos = {'Block': (.5+rw*bf*0, 4),

                'Segment': (.5+rw*bf*1, .5),

                'Event': (.5+rw*bf*4, 6),
                'EventArray': (.5+rw*bf*4, 4),
                'Epoch': (.5+rw*bf*4, 2),
                'EpochArray': (.5+rw*bf*4, .2),

                'RecordingChannelGroup': (.5+rw*bf*.8, 8.5),
                'RecordingChannel': (.5+rw*bf*1.2, 5.5),

                'Unit': (.5+rw*bf*2., 9.5),

                'SpikeTrain': (.5+rw*bf*3, 9.5),
                'Spike': (.5+rw*bf*3, 7.5),

                'IrregularlySampledSignal': (.5+rw*bf*3, 4.9),
                'AnalogSignal': (.5+rw*bf*3, 2.7),
                'AnalogSignalArray': (.5+rw*bf*3, .5),
                }
    generate_diagram('simple_generated_diagram.svg',
                     rect_pos, rect_width, figsize)
    generate_diagram('simple_generated_diagram.png',
                     rect_pos, rect_width, figsize)


if __name__ == '__main__':
    generate_diagram_simple()
    pyplot.show()
