#! /usr/bin/env python

import os
import io
from distutils.core import setup
# from setuptools import setup

# NOTE: to compile, run in the current directory
# python setup.py build_ext --inplace
# python setup.py develop


def find_all_package_directories(packages, package_name, current_dir):
    for dirname in os.listdir(current_dir):
        if os.path.isfile(os.path.join(current_dir, dirname, '__init__.py')):
            package_n = '{}.{}'.format(package_name, dirname)
            packages += [package_n]
            find_all_package_directories(packages, package_n, os.path.join(current_dir, dirname))


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


here = os.path.abspath(os.path.dirname(__file__))

long_description = read('README.md', 'CHANGELOG.md')

python_dir = os.path.join(here, 'magnolia', 'python')

packages = []
find_all_package_directories(packages, 'magnolia', python_dir)
# packages = ['magnolia']
# packages = ['magnolia.{}'.format(name) for name in os.listdir(python_dir) if os.path.isdir(os.path.join(python_dir, name))]
# packages += ['magnolia.python.models']
# package_dirs={}
# for package in packages:
#     package_dirs[package] = os.path.join('magnolia', 'python')


setup(
    name='magnolia',
    description='Audio source separation and denoising',
    long_description=long_description,
    maintainer='Lab41',
    url='https://github.com/lab41/magnolia',
    # author_email='jhetherly@iqt.org',
    license='MIT',
    platforms='any',

    # packages=packages,
    # package_dir=package_dirs,
    # packages=['magnolia'],
    # package_dir={'': 'python'},
    package_dir={'magnolia': os.path.join('magnolia', 'python')},
    packages=packages,

    # Required packages
    install_requires=[
       'matplotlib',
       'numpy',
       'pandas',
       'scipy',
       'scikit-learn',
       'seaborn',
       'tqdm',
       'msgpack-python',
    ],
    version = '0.2.0'
)
