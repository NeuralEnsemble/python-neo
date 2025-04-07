"""
Common tools that are useful for neo.io object tests
"""

import logging
import os
import importlib.util


logger = logging.getLogger("neo.test")


def can_use_network():
    """
    Return True if network access is allowed
    """

    # env variable for local dev
    if os.environ.get("NEO_TESTS_NO_NETWORK", False):
        return False

    # check for datalad presence
    datalad_spec = importlib.util.find_spec("datalad")
    if datalad_spec is not None:
        HAVE_DATALAD = True
    else:
        HAVE_DATALAD = False

    if not HAVE_DATALAD:
        return False

    return True
