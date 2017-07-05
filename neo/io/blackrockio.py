

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.blackrockrawio import BlackrockRawIO

class BlackrockIO(BlackrockRawIO, BaseFromRaw):
    name = 'Blackrock IO'
    description = "This IO reads .nev/.nsX file of the Blackrock " + \
        "(Cerebus) recordings system."
    
    __prefered_signal_group_mode = 'split-all'
    
    def __init__(self, filename, nsx_to_load=None, **kargs):
        BlackrockRawIO.__init__(self, filename=filename, nsx_to_load=nsx_to_load, **kargs)
        BaseFromRaw.__init__(self, filename)
        




