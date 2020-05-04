__author__ = 'Peter Komar "\
             "<peter.komar@totient.bio>'
__copyright__ = '2020 Totient, Inc'
__version__ = '2.0.0'

import io
from datetime import datetime
from setuptools import setup, find_packages

setup(
    name='deldenoiser',
    version=__version__,
    description='Denoise sequencing data from DEL screens.',
    long_description=io.open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX',
    ],
    author='Peter Komar',
    author_email='peter.komar@totient.bio',
    maintainer='Peter Komar',
    maintainer_email='peter.komar@totient.bio',
    url='https://github.com/totient-bio/deldenoiser',
    license='Copyright (c) {} Totient, Inc.'.format(
        datetime.now().year
    ),
    packages=find_packages(),
    install_requires=io.open('requirements.txt').read().splitlines(),
    include_package_data=True,
    scripts=["command-line-tool/deldenoiser"],
    python_requires='>=3.6'
)
