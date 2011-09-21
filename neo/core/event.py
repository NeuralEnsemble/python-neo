from neo.core.baseneo import BaseNeo

import quantities as pq


class Event(BaseNeo):
    """
    Object to represent an event occurring at a particular time.
    Useful for managing trigger, stimulus, ...
    
    *Usage*:
    
    *Required attributes/properties*:
        :time: (quantity):
        :label: (str): 
    
    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:    
    """
    def __init__(self, time, label, **kargs):
        BaseNeo.__init__(self, **kargs)
        self.time = time
        self.label = label
