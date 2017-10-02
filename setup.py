#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import os

long_description = open("README.rst").read()
install_requires = ['numpy>=1.7.1',
                    'quantities>=0.9.0']
extras_require = {
    'hdf5io': ['h5py'],
    'igorproio': ['igor'],
    'kwikio': ['scipy', 'klusta'],
    'neomatlabio': ['scipy>=0.12.0'],
    'nixio': ['nixio'],
    'stimfitio': ['stfio'],
}

if os.environ.get('TRAVIS') == 'true' and \
    os.environ.get('TRAVIS_PYTHON_VERSION').startswith('2.6'):
    install_requires.append('unittest2>=0.5.1')

with open("neo/version.py") as fp:
    d = {}
    exec(fp.read(), d)
    neo_version = d['version']

setup(
    name = "neo",
    version = neo_version,
    packages = ['neo', 'neo.core', 'neo.io', 'neo.test', 'neo.test.iotest', 
                'neo.rawio', 'neo.rawio.tests'],
    install_requires=install_requires,
    extras_require=extras_require,
    author = "Neo authors and contributors",
    author_email = "samuel.garcia@cnrs.fr",
    description = "Neo is a package for representing electrophysiology data in Python, together with support for reading a wide range of neurophysiology file formats",
    long_description = long_description,
    license = "BSD-3-Clause",
    url='http://neuralensemble.org/neo',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering']
)
