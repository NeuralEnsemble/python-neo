# encoding: utf-8
"""
This generate diagram in .png and .svg from neo.description

TODO:
many_to_many_relationship in matplotlib style annotation.

Author: sgarcia
"""

from matplotlib import pyplot
from matplotlib.patches import Rectangle , ArrowStyle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

from neo.description import class_by_name, one_to_many_reslationship, many_to_many_reslationship, \
        classes_necessary_attributes, classes_recommended_attributes

import quantities as pq
from datetime import datetime
import numpy as np

line_heigth = .22
fontsize = 10.5
left_text_shift = .1


def generate_diagram(filename, rect_pos,rect_width,  figsize ):
    rw = rect_width
    
    fig = pyplot.figure(figsize = figsize)
    ax = fig.add_axes([0,0,1,1])    
    
    #calculate height
    all_h = { }
    for name in rect_pos.keys():
        # rectangles
        htotal = (1.5+len(classes_necessary_attributes[name]) + len(classes_recommended_attributes[name]))*line_heigth
        if name in one_to_many_reslationship: htotal += len(one_to_many_reslationship[name])*line_heigth
        if name in many_to_many_reslationship: htotal += len(many_to_many_reslationship[name])*line_heigth
        all_h[name] = htotal
            

    # draw connections
    for name in rect_pos.keys():
        #~ pos = rect_pos[name]
        #~ htotal = all_h[name]
        #~ if name not in one_to_many_reslationship.keys(): continue
        relationship = [ ]
        if name in one_to_many_reslationship: relationship += one_to_many_reslationship[name]
        #~ if name in many_to_many_reslationship: relationship += many_to_many_reslationship[name]
        for c,children in enumerate(relationship):
            if children not in rect_pos.keys(): continue
            x = rect_pos[children][0] 
            y = rect_pos[children][1] + all_h[children] - line_heigth*.5
            x2 = rect_pos[name][0] + rect_width
            y2 = rect_pos[name][1] + all_h[name] - line_heigth*1.5 - line_heigth*c
            a = ax.annotate('', (x, y),
                                        (x2,y2),
                                        #xycoords="figure fraction", textcoords="figure fraction",
                                        ha="right", va="center",
                                        size=fontsize,
                                        arrowprops=dict(arrowstyle='fancy',
                                                        #~ patchB=p,
                                                        shrinkA=.3,
                                                        shrinkB=.3,
                                                        fc="w", ec="k",
                                                        connectionstyle="arc3,rad=-0.05",
                                                        ),
                                        bbox=dict(boxstyle="square", fc="w"))
            a.set_zorder(-4)

    
    # draw boxes
    for name in rect_pos.keys():
        pos = rect_pos[name]
        htotal = all_h[name]
        relationship = [ ]
        if name in one_to_many_reslationship: relationship += one_to_many_reslationship[name]
        if name in many_to_many_reslationship: relationship += many_to_many_reslationship[name]
        attributes = classes_necessary_attributes[name]+classes_recommended_attributes[name]        

        rect = Rectangle(pos,rect_width ,htotal,
                                    facecolor = 'w',
                                    edgecolor = 'k',
                                    linewidth = 2.,
                                    )
        ax.add_patch(rect)
        
        # title green
        pos2 = pos[0] , pos[1]+htotal - line_heigth*1.5
        rect = Rectangle(pos2,rect_width ,line_heigth*1.5,
                                    facecolor = 'g',
                                    edgecolor = 'k',
                                    alpha = .5,
                                    linewidth = 2.,
                                    )
        ax.add_patch(rect)
        
        #relationship
        pos2 = pos[0] , pos[1]+htotal - line_heigth*(1.5+len(relationship))
        rect = Rectangle(pos2,rect_width ,line_heigth*len(relationship),
                                    facecolor = 'c',
                                    edgecolor = 'k',
                                    alpha = .5,
                                    )
        ax.add_patch(rect)
        
        # necessary attr
        pos2 = pos[0] , pos[1]+htotal - line_heigth*(1.5+len(relationship)+len(classes_necessary_attributes[name]))
        rect = Rectangle(pos2,rect_width ,line_heigth*len(classes_necessary_attributes[name]),
                                    facecolor = 'r',
                                    edgecolor = 'k',
                                    alpha = .5,
                                    )
        ax.add_patch(rect)
        

        
        # name
        ax.text( pos[0]+rect_width/2. , pos[1]+htotal - line_heigth*1.5/2. , name,
                            horizontalalignment = 'center',
                            verticalalignment = 'center',
                            fontsize = fontsize+2,
                            fontproperties = FontProperties(weight = 'bold', 
                                                                                ),
                    )
        #relationship
        for i,relat in enumerate(relationship):
            ax.text( pos[0]+left_text_shift , pos[1]+htotal - line_heigth*(i+2),
                                relat.lower()+'s: list',
                                horizontalalignment = 'left',
                                verticalalignment = 'center',
                                fontsize = fontsize,
                        )
        # attributes
        for i,attr in enumerate(attributes):
            attrname, attrtype = attr[0], attr[1]
            t1 = attrname
            if attrname =='': t1 = 'Inherited'
            else: t1 = attrname
                
            if attrtype == pq.Quantity:
                if attr[2] == 0: t2 = 'Quantity scalar'
                else: t2 = 'Quantity %dD'%attr[2]
            elif attrtype == np.ndarray: t2 = "np.ndarray %dD dt='%s'"%(attr[2], attr[3].kind)
            elif attrtype == datetime: t2 = 'datetime'
            else:t2 = attrtype.__name__
            
            t  = t1+' :  '+t2
            ax.text( pos[0]+left_text_shift , pos[1]+htotal - line_heigth*(i+len(relationship)+2), 
                                t,
                                horizontalalignment = 'left',
                                verticalalignment = 'center',
                                fontsize = fontsize,
                        )
        
        
    xlim, ylim = figsize
    ax.set_xlim(0,xlim)
    ax.set_ylim(0,ylim)

    ax.set_xticks([ ])
    ax.set_yticks([ ])
    fig.savefig(filename)



def generate_diagram_simple():
    figsize = (18, 12)
    rw = rect_width = 3.
    bf = blank_fact = 1.2
    rect_pos = {
                    'Block' : (.5+rw*bf*0,4),
                    
                    'Segment' : ( .5+rw*bf*1, 3),

                    'Event': ( .5+rw*bf*4, 6),
                    'EventArray': ( .5+rw*bf*4, 4),
                    'Epoch': ( .5+rw*bf*4, 2),
                    'EpochArray': ( .5+rw*bf*4, .2),

                    'RecordingChannelGroup': ( .5+rw*bf*1, .5 ),
                    'RecordingChannel': ( .5+rw*bf*2, 4 ),

                    'Unit': ( .5+rw*bf*2, 7),
                    
                    'SpikeTrain': ( .5+rw*bf*3, 9.5),
                    'Spike': ( .5+rw*bf*3, 7.5),
                    
                    'IrregularlySampledSignal': ( .5+rw*bf*3, 4.9),
                    'AnalogSignal': ( .5+rw*bf*3, 2.7),
                    'AnalogSignalArray': ( .5+rw*bf*3, .5),
                    
                    
                    }
    generate_diagram('simple_generated_diagram.svg', rect_pos, rect_width, figsize)
    generate_diagram('simple_generated_diagram.png', rect_pos, rect_width, figsize)
    


if __name__ == '__main__':
    
    generate_diagram_simple()
    pyplot.show()


