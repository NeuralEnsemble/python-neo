#!/usr/bin/env python

from setuptools import setup, find_packages
import os

long_description = open("README.rst").read()
install_requires = ['packaging',
                    'numpy>=1.18.5',
                    'quantities>=0.12.1']
extras_require = {
    'igorproio': ['igor'],
    'kwikio': ['klusta'],
    'neomatlabio': ['scipy>=1.0.0'],
    'nixio': ['nixio>=1.5.0'],
    'stimfitio': ['stfio'],
    'tiffio': ['pillow'],
    'edf': ['pyedflib'],
    'ced': ['sonpy'],
    'nwb': ['pynwb'],
    'maxwell': ['h5py'],
    'biocam': ['h5py'],
}
extras_require["all"] = sum(extras_require.values(), [])

# explicitly removing stfio from list of all as it is not pip installable
extras_require["all"].remove('stfio')

with open("neo/version.py") as fp:
    d = {}
    exec(fp.read(), d)
    neo_version = d['version']

setup(
    name="neo",
    version=neo_version,
    packages=find_packages(),
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
    python_requires=">=3.7",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering']
)
