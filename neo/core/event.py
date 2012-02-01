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
    def __init__(self, time, label, name=None, description=None,
                 file_origin=None, **annotations):
        """Initialize a new Event."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        self.time = time
        self.label = label
        
        self.segment =None
