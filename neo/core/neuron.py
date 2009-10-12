# -*- coding: utf-8 -*-

global id
id = 0

class Neuron(object):
     
    """
    Stores properties that define a neuron

    **Definition**
    A :class:`Neuron` regroups all the :class:`SpikeTrain` objects within a common
    :class:`Block`, gathered accross several :class:`Segment`, that has been emitted
    by the same cell.

    with arguments:
    
    ``id``
        A number identifying the neuron. A default number is created if none are given
    
    ``label``
        A string describing the neuron

    **Usage**

    **Example**
    
    """
    label = None
    id    = None
    
    def __init__(self, *arg, **karg):
        global id
        self.spiketrains = []
        
        if karg.has_key('id'):
            self.id = karg['id']            
        else:
            self.id = id
            id += 1
                
        if karg.has_key('label'):
            self.label = karg['label']
        else:
            self.label = "Neuron %d" %self.id
        
    def __str__(self):
        return str(self.label) 
    
    def add_spiketrain(self, spiketrain):
        self.spiketrains += [spiketrain]