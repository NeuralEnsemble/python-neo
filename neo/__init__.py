# -*- coding: utf-8 -*-
'''
Neo is a package for representing electrophysiology data in Python,
together with support for reading a wide range of neurophysiology file formats
'''

from neo.core import *
from neo.io import *
from neo.version import version as __version__

import logging

try:  # NullHandler was introduced in Python 2.7
    _NullHandler = logging.NullHandler
except AttributeError:
    class _NullHandler(logging.Handler):
        def emit(self, record):
            pass

# No logging by default, but suppress error message about missing handler.
log = logging.getLogger(__name__)
log.addHandler(_NullHandler())
