from neo.core.baseneo import BaseNeo

import numpy as np
import quantities as pq

class IrregularlySampledSignal(BaseNeo):
    """
    Object to manage signal when the sampling is not regular.
    In short this object manage both the signal values and its times vector.
    
    *Usage*:
        TODO
    
    *Required attributes/properties*:
        :times: Quantitiy vector
        :values: Quantitiy vector same size of times
    
    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:                
    """
    def __init__(self, times, values,
                  name=None, description=None,
                 file_origin=None, **annotations):
        """Initalize a new IrregularlySampledSignal."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        
        self.times = times
        self.values = values
        
        self.segment = None
        self.recordingchannel = None
        
