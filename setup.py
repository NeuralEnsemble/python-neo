#!/usr/bin/env python

from setuptools import setup
import os

long_description = open("README.rst").read()
install_requires = ['numpy>=1.11.0,!=1.16.0',
                    'quantities>=0.12.1']
extras_require = {
    'hdf5io': ['h5py'],
    'igorproio': ['igor'],
    'kwikio': ['scipy', 'klusta'],
    'neomatlabio': ['scipy>=0.12.0'],
    'nixio': ['nixio>=1.5.0b2'],
    'stimfitio': ['stfio'],
    'tiffio': ['pillow']
}

with open("neo/version.py") as fp:
    d = {}
    exec(fp.read(), d)
    neo_version = d['version']

setup(
    name="neo",
    version=neo_version,
    packages=[
        'neo', 'neo.core', 'neo.io', 'neo.rawio', 'neo.test',
        'neo.test.coretest', 'neo.test.iotest', 'neo.test.rawiotest'],
    install_requires=install_requires,
    extras_require=extras_require,
    author="Neo authors and contributors",
    author_email="samuel.garcia@cnrs.fr",
    description="Neo is a package for representing electrophysiology data in "
                "Python, together with support for reading a wide range of "
                "neurophysiology file formats",
    long_description=long_description,
    license="BSD-3-Clause",
    url='https://neuralensemble.org/neo',
    python_requires=">=3.5",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering']
)
