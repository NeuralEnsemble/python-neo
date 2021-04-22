"""
Common tools that are useful for neo.io object tests
"""

import logging
import os
import tempfile

from urllib.request import urlopen

logger = logging.getLogger("neo.test")


def can_use_network():
    """
    Return True if network access is allowed
    """
    if os.environ.get('NOSETESTS_NO_NETWORK', False):
        return False
    if os.environ.get('TRAVIS') == 'true':
        return False
    return True
