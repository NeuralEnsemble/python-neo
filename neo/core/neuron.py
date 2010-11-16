# -*- coding: utf-8 -*-

from baseneo import BaseNeo

global neoid
neoid = 0

class Neuron(object):
    definition = """A :class:`Neuron` regroups all the :class:`SpikeTrain` objects within a common
    :class:`Block`, gathered accross several :class:`Segment`, that has been emitted
    by the same cell.

    """
     
    __doc__ = """
    Stores properties that define a neuron

    **Definition**"""+definition+"""

    with arguments:
    
    ``neoid``
        A number identifying the neuron. A default number is created if none are given
    
    ``label``
        A string describing the neuron

    **Usage**

    **Example**
    
    """
    
    def __init__(self, *arg, **karg):
        global neoid
        self._spiketrains = []
        
        if karg.has_key('neoid'):
            self.neoid = karg['neoid']            
        else:
            self.neoid = neoid
            neoid += 1
                
        if karg.has_key('name'):
            self.name = karg['name']
        else:
            self.name = "Neuron %d" %self.neoid
        
    
    def add_spiketrain(self, spiketrain):
        self._spiketrains += [spiketrain]
