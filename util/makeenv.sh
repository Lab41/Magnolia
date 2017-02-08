#!/bin/bash

# install python2 and utilities
sudo chmod -R a+wrx /opt/conda
conda create -n python2 python=2 ipykernel
source activate python2
python -m ipykernel install --user
conda info --envs
source activate python2
pip install numpy
pip install scipy
pip install ipython
pip install scikits.audiolab
pip install matplotlib
pip install glob
pip install sklearn
source deactivate

# install swiss army knife sound utility
sudo apt-get update
sudo apt-get install sox


