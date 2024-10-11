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

    if os.environ.get("NEO_TESTS_NO_NETWORK", False):
        return False
    try:
        import datalad

        HAVE_DATALAD = True
    except:
        HAVE_DATALAD = False
    if not HAVE_DATALAD:
        return False
    return True
