class BaseNeo(object):
    metadata = {}

    def __setattr__(self, k, v):
        self.metadata[k] = v
        self.__dict__[k] = v

    def __delattr__(self, k):
        del self.metadata[k]
        del self.__dict__[k]
