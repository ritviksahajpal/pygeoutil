from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "pygeoutil: collection of plotting and geographic utility functions"
# Long description will go up on the pypi page
long_description = """

pygeoutil
========
pygeoutil is a collection of python utilities to:
    1. Plot data
    2. Utilities related to netCDF data
    3. Higher-level utilities to handle geographic data

License
=======
``pygeoutil`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2015--, Ritvik Sahajpal, University of Maryland College Park
Department of Geographical Sciences
"""

NAME = "pygeoutil"
MAINTAINER = "Ritvik Sahajpal"
MAINTAINER_EMAIL = "ritvik@umd.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/ritviksahajpal/pygeoutil.git"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Ritvik Sahajpal"
AUTHOR_EMAIL = "ritvik@umd.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGES = ['pygeoutil',
            'pygeoutil.tests']
PACKAGE_DATA = {'pygeoutil': [pjoin('data', '*')]}
REQUIRES = ["numpy"]
