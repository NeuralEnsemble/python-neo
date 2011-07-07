# -*- coding: utf-8 -*-


class Epoch(object):
    definition = """Similar to Event but with a duration. Useful for 
    describing a period, the state of a subject, ..."""
     
    __doc__ = """
    Object to represent a period of time.

    **Definition**
    %s
    
    with arguments:    
        ``time`` The starting time of the epoch    
        ``duration`` The duration of the epoch    
        ``label``
    
    **Usage**
    # Create an Epoch, add to Segment
    e = Epoch(time=3.3, duration=1.0, label='Explosion')
    seg._events.append(e)
    print seg.get_events()[0].label    
    """ % definition
    
    time     = None
    label    = None 
    duration = 0
    
    def __init__(self, *arg, **karg):
        """Initialize an Epoch and store the following parameters:
            time
            label
            duration
        """
    
        if 'time' in karg.keys():
            self.time = karg['time']
        
        if 'label' in karg.keys():
            self.label = karg['label']
            
        if 'duration' in karg.keys():
            self.duration = karg['duration']
    
    #~ def __str__(self):
        #~ res = "Epoch %s at time %g with duration %g" %(self.label, self.time, self.duration)
        #~ return res
    