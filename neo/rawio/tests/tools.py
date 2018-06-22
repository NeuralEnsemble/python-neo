# -*- coding: utf-8 -*-
"""
Common tools that are useful for neo.io object tests
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import logging
import os
import shutil
import tempfile

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen


def can_use_network():
    """
    Return True if network access is allowed
    """
    if os.environ.get('NOSETESTS_NO_NETWORK', False):
        return False
    if os.environ.get('TRAVIS') == 'true':
        return False
    return True


def make_all_directories(filename, localdir):
    """
    Make the directories needed to store test files
    """
    # handle case of multiple filenames
    if not hasattr(filename, 'lower'):
        for ifilename in filename:
            make_all_directories(ifilename, localdir)
        return

    fullpath = os.path.join(localdir, os.path.dirname(filename))
    if os.path.dirname(filename) != '' and not os.path.exists(fullpath):
        if not os.path.exists(os.path.dirname(fullpath)):
            make_all_directories(os.path.dirname(filename), localdir)
        os.mkdir(fullpath)


def download_test_file(filename, localdir, url):
    """
    Download a test file from a server if it isn't already available.

    filename is the name of the file.

    localdir is the local directory to store the file in.

    url is the remote url that the file should be downloaded from.
    """
    # handle case of multiple filenames
    if not hasattr(filename, 'lower'):
        for ifilename in filename:
            download_test_file(ifilename, localdir, url)
        return

    localfile = os.path.join(localdir, filename)
    distantfile = url + '/' + filename

    if not os.path.exists(localfile):
        logging.info('Downloading %s here %s', distantfile, localfile)
        dist = urlopen(distantfile)
        with open(localfile, 'wb') as f:
            f.write(dist.read())


def create_local_temp_dir(name, directory=None):
    """
    Create a directory for storing temporary files needed for testing neo

    If directory is None or not specified, automatically create the directory
    in {tempdir}/files_for_testing_neo on linux/unix/mac or
    {tempdir}\files_for_testing_neo on windows, where {tempdir} is the system
    temporary directory returned by tempfile.gettempdir().
    """
    if directory is None:
        directory = os.path.join(tempfile.gettempdir(),
                                 'files_for_testing_neo')

    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = os.path.join(directory, name)
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory
