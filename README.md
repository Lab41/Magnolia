# Magnolia

We will focus on three independent problems. These are:

1. **Monaural source separation** - 
   Using [source contrastive estimation](https://arxiv.org/abs/1705.04662) (submitted to SiPS 2017).
2. Removing a source (whose location is static) that is dynamic and loud.
3. Multiple moving sources (i.e., the channel changes with time) 
4. Standoff distance acoustic enhancement (dynamic gain adjustment in low SNR)

This project will be primarily an analysis of cost functions (in supervised and unsupervised settings) that can be used in order to denoise and isolate signals. The resulting algorithms of interest will be a mix of cost function resembling ICA and simulated beamforming methods. 

## Installation

If you'd like to just call the code in the repository:
```
pip install --upgrade --no-deps git+https://github.com/lab41/magnolia
```

To update our files and edit our code as well as look at our notebooks:
```
git clone https://github.com/lab41/magnolia
pip install --upgrade --no-deps magnolia/
```
Our notebooks are located at:
```
magnolia/src/notebooks
```

## Demonstration

### Source Separation

Source separation is currently on a branch that will not run in the current code base. We have released a previous version (release 1) that does accomodate for it. This can be run out of the box with two shell scripts:

1. `source-separation-build.sh` : Per the above comment, this will checkout the release that is able to run source separtion, provided you have cloned the original repository. Then it will compile a Dockerfile (that resembles the one currently in the repository.) This will download models from Dropbox, which take the most amount of time. Then, it will revert back to the master. There might be some tricks you need to do to get yourself back to the right git state. The docker container is called magnolia-demo-r1, for the first release.

2. `source-separation-run.sh` : This will actually run magnolia-demo-r1.sh. After you type in this command, go to your localhost at port 5000. (Type in `localhost:5000` into your browser.)

## Subdirectory contents:

### Data
- Directory: [`data`](https://github.com/Lab41/Magnolia/tree/master/data)
- Contents: Data.

### Documentation
- Directory: [`docs`](https://github.com/Lab41/Magnolia/tree/master/docs)
- Contents: Documentation, including initial pitch and planning.

### Source Code 
- Directory: [`src`](https://github.com/Lab41/Magnolia/tree/master/src)
- Contents: Source code.

### Sandbox
- Directory: [`sandbox`](https://github.com/Lab41/Magnolia/tree/master/sandbox)
- Contents: Code we used to play around with certain algorithms

### Utility
- Directory: [`util`](https://github.com/Lab41/Magnolia/tree/master/util)
- Contents: Utility and reference, including required installation packages

### Evaluation
- Directory: [`evaluation`](https://github.com/Lab41/Magnolia/tree/master/evaluation)
- Contents: Evaluation code
