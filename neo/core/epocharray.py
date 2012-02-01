from neo.core.baseneo import BaseNeo

import numpy as np
import quantities as pq

class EpochArray(BaseNeo):
    """
    Array of epochs. Introduced for performance reason.
    An :class:`EpochArray` is prefered to a list of :class:`Epoch` objects.
    
    *Usage*:
        TODO
    
    *Required attributes/properties*:
        :times: (quantity array 1D)
        :durations: (quantity array 1D)
        :labels: (numpy.array 1D dtype='S') )
    
    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:         
    """
    def __init__(self, times=np.array([])*pq.s, durations=np.array([])*pq.s,
                 labels=np.array([], dtype='S'), name=None, description=None,
                 file_origin=None, **annotations):
        """Initialize a new EpochArray."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        
        self.times = times
        self.durations = durations
        self.labels = labels
        
        self.segment =None
