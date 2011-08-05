# encoding: utf-8
"""
This generate diagram in .png and .svg from neo.description

"""
from matplotlib import pyplot
from matplotlib.patches import Rectangle , ArrowStyle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

from neo.description import classnames, one_to_many_reslationship, many_to_many_reslationship, \
        classes_necessary_attributes, classes_recommended_attributes

import quantities as pq
from datetime import datetime
import numpy as np

line_heigth = .2
fontsize = 10
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
            
    
    #~ # draw connections
    #~ for name,OEclass in allclasses.iteritems():
        #~ if name in rect_pos:
            #~ pos = rect_pos[name]
            #~ htotal = all_h[name]
            
            #~ for i,parentname in enumerate(OEclass.parents):
                #~ if parentname in rect_pos:
                    
                    #~ x = rect_pos[parentname][0] + rect_width
                    #~ y = rect_pos[parentname][1] + all_h[parentname] - line_heigth*2.
                    #~ x2 = rect_pos[name][0]
                    #~ y2 = rect_pos[name][1] + htotal - line_heigth*3 - line_heigth*i 
                    #~ dx = x2- x
                    #~ dy = y2 - y
                    
                    
                    #~ arrow = FancyArrowPatch(
                                                #~ posA = (x2,y2) , posB =(x, y) ,
                                                
                                                #~ facecolor = 'w',
                                                #~ linewidth = 2,
                                                
                                                #~ arrowstyle = 'wedge, tail_width=0.3,shrink_factor=0.3',
                                                #~ #arrowstyle = "Fancy, head_length=.4, head_width=.4, tail_width=.4",
                                                #~ #arrowstyle = '-[',
                                                
                                                
                                                #~ connectionstyle='arc3, rad=-.1',
                                                
                                                #~ mutation_scale=25,
                                                #~ )
                                                
                    #~ ax.add_patch(arrow)
                    
                    #~ #ax.arrow(x,y,dx,dy,
                    #~ #                zorder = -4,
                    #~ #                shape = 'full',
                    #~ #                length_includes_head = True,
                    #~ #                )



    
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
        
        pos2 = pos[0] , pos[1]+htotal - line_heigth*1.5
        rect = Rectangle(pos2,rect_width ,line_heigth*1.5,
                                    facecolor = 'g',
                                    edgecolor = 'k',
                                    alpha = .3,
                                    linewidth = 2.,
                                    )
        ax.add_patch(rect)
        
        pos2 = pos[0] , pos[1]+htotal - line_heigth*(1.5+len(relationship))
        rect = Rectangle(pos2,rect_width ,line_heigth*len(relationship),
                                    facecolor = 'y',
                                    edgecolor = 'k',
                                    alpha = .1,
                                    )
        ax.add_patch(rect)

        #~ pos2 = pos[0] , pos[1]+htotal - line_heigth*(2.5+len(OEclass.parents))
        #~ h = line_heigth*len(OEclass.parents)
        #~ rect = Rectangle(pos2,rect_width ,h,
                                    #~ facecolor = 'm',
                                    #~ edgecolor = 'k',
                                    #~ alpha = .1,
                                    #~ )
        #~ ax.add_patch(rect)
        
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
                                '_'+relat+'s',
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
            elif attrtype == np.array: t2 = "np.array %dD dt='%s'"%(attr[3], attr[2].kind)
            elif attrtype == datetime: t2 = 'datetime'
            else:t2 = str(attrtype)
                
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
    filename = 'simple_diagram.svg'
    figsize = (18, 12)
    rw = rect_width = 2.8
    bf = blank_fact = 1.2
    rect_pos = {
                    'block' : (.5+rw*bf*0,4),
                    
                    'segment' : ( .5+rw*bf*1, 3),

                    'event': ( .5+rw*bf*4, 6),
                    'eventarray': ( .5+rw*bf*4, 4),
                    'epoch': ( .5+rw*bf*4, 2),
                    'epocharray': ( .5+rw*bf*4, .5),

                    'recordingchannelgroup': ( .5+rw*bf*1, .5 ),
                    'recordingchannel': ( .5+rw*bf*2, 4 ),

                    'unit': ( .5+rw*bf*2, 7),
                    
                    'spiketrain': ( .5+rw*bf*3, 9.3),
                    'spike': ( .5+rw*bf*3, 7.5),
                    
                    'irsaanalogsignal': ( .5+rw*bf*3, 4.9),
                    'analogsignal': ( .5+rw*bf*3, 2.7),
                    'analogsignalarray': ( .5+rw*bf*3, .5),
                    
                    
                    }
    generate_diagram(filename, rect_pos, rect_width, figsize)


if __name__ == '__main__':
    
    generate_diagram_simple()
    pyplot.show()


