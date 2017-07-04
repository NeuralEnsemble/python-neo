

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.blackrockrawio import BlackrockRawIO

class BlackrockIO(BlackrockRawIO, BaseFromRaw):
    name = 'Blackrock IO'
    description = "This IO reads .nev/.nsX file of the Blackrock " + \
        "(Cerebus) recordings system."
    
    def __init__(self, **kargs):
        BlackrockRawIO.__init__(self, **kargs)
        BaseFromRaw.__init__(self, **kargs)
        




