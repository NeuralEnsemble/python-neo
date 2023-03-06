"""
Common tools that are useful for neo.io object tests
"""

import logging
import os

logger = logging.getLogger("neo.test")


def can_use_network():
    """
    Return True if network access is allowed
    """
    try:
        import datalad
        HAVE_DATALAD = True
    except:
        HAVE_DATALAD = False
    if not HAVE_DATALAD:
        return False
    if os.environ.get('NOSETESTS_NO_NETWORK', False):
        return False
    if os.environ.get('TRAVIS') == 'true':
        return False
    return True
