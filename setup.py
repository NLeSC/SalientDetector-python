# -*- coding: utf-8 -*-
'''
Setting up the Salient region detection in images package.
'''
from setuptools import setup, find_packages

setup(
    name='salientregions',
    version='1.0.0',
    description='Package for finding salient regions in images',
    #long_description=readme,
    author='Netherlands eScience Center',
    url='https://github.com/NLeSC/SalientRegions-python',
    download_url = 'https://github.com/NLeSC/SalientDetector-python/tarball/v1.0.0',
    packages=find_packages(exclude=('tests'))
)
