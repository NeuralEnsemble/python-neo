# -*- coding: utf-8 -*-
from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.tdtrawio import TdtRawIO


class TdtIO(TdtRawIO, BaseFromRaw):
    """
    Class for reading data from from Tucker Davis TTank format.

    Terminology:
    TDT holds data with tanks (actually a directory). And tanks hold sub blocks
    (sub directories).
    Tanks correspond to Neo Blocks and TDT blocks correspond to Neo Segments.
    """
    _prefered_signal_group_mode = 'split-all'
    mode = 'dir'

    def __init__(self, dirname):
        TdtRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)
