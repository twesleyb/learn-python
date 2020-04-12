#!/usr/bin/env python3

# From: https://www.codementor.io/@arpitbhayani/host-your-python-package-using-github-on-pypi-du107t7ku

# Create PyPi Live and PyPi Test accounts. 

# The .pypirc contains your PyPi configuration:

'''.pypirc

[distutils]
index-servers =
  pypi
    pypitest

    [pypi]
    repository: https://pypi.python.org/pypi
    username: YOUR_USERNAME_HERE
    password: YOUR_PASSWORD_HERE

    [pypitest]
    repository: https://testpypi.python.org/pypi
    username: YOUR_USERNAME_HERE
    password: YOUR_PASSWORD_HERE
'''

# Create your package. An example directory structure:
'''project-directory
source_dir/                 # the source directory
|-- my_python_package       # your package
|   |-- __init__.py
|   `-- FILES ....          # your package files
|-- README.md
|-- setup.cfg
|-- setup.py
'''

# host your package on github.

# Release your package:
# 1. Go to project homepage
# 2. Click on the Release link.
# 3. Click draft a new release.
# 4. Fill out the prompt.
# 5. Click publish.
# 6. Copy the download link (tar.gz) and save it onto your pc.

# Edit setup.py:
'''
from distutils.core import setup

setup(
    name = 'my_python_package',
    packages = ['my_python_package'],
    version = 'version number',  # Ideally should be same as your GitHub release tag varsion
    description = 'description',
    author = '',
    author_email = '',
    url = 'github package source url',
    download_url = 'download link you saved',
    keywords = ['tag1', 'tag2'],
    classifiers = [],
)
'''

# Edit setup.cfg
'''
[metadata]
description-file = README.md
'''

