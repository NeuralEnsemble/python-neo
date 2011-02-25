from neo.core.baseneo import BaseNeo

class Block(BaseNeo):
    def __init__(self, name='', filedatetime=None, index=None, segments=[]):
        self.name = name
        self.filedatetime = filedatetime
        self.index = index
        self.segments = segments
        BaseNeo.__init__(self)
