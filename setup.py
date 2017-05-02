"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='magnolia',

    description='Audio source separation and denoising',
    long_description=long_description,

    maintainer='Lab41',

    # The project's main homepage.
    url='https://github.com/lab41/magnolia',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    package_dir={'magnolia': 'src'},
    packages=['magnolia','magnolia.features','magnolia.factorization','magnolia.utils','magnolia.dnnseparate'],
    #packages=['features','factorization','utils'],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    # py_modules=["magnolia"],

    # Required packages
    install_requires=[
       'matplotlib',
       'numpy',
       'python_speech_features',
       'scipy',
       'scikit-learn',
       'seaborn',
       'soundfile',
    ],

    version = '0.1.0'

)
