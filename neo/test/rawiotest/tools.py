"""
Common tools that are useful for neo.io object tests
"""

import logging
import os

from neo.utils import HAVE_DATALAD

logger = logging.getLogger("neo.test")


def can_use_network():
    """
    Return True if network access is allowed
    """
    if not HAVE_DATALAD:
        return False
    if os.environ.get('NOSETESTS_NO_NETWORK', False):
        return False
    if os.environ.get('TRAVIS') == 'true':
        return False
    return True
