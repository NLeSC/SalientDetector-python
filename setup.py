# -*- coding: utf-8 -*-
'''
Setting up the Salient region detection in images package.
'''
from setuptools import setup, find_packages
from pip.req import parse_requirements


with open('README.md') as f:
    readme = f.read()

# Parse requirements.txt file
lines = parse_requirements('requirements.txt', session=False)
requirements = [str(item.req) for item in lines]

setup(
    name='salientregions',
    version='1.0.0',
    install_requires=requirements,
    description='Package for finding salient regions in images',
    #long_description=readme,
    author='Netherlands eScience Center',
    url='https://github.com/NLeSC/SalientRegions-python',
    download_url = 'https://github.com/NLeSC/SalientDetector-python/tarball/v1.0.0',
    packages=find_packages(exclude=('tests'))
)
