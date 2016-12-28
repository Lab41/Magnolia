# Magnolia

We will focus on three independent problems. These are:

1. Removing a source (whose location is static) that is dynamic and loud
2. Multiple moving sources (i.e., the channel changes with time)
3. Standoff distance acoustic enhancement (dynamic gain adjustment in low SNR)

The resulting algorithms of interest will be a mix of ICA and simulated beamforming methods.

## Dataset and Collection

Sampling rate at 48kHz, but we can apply anti-aliasing and subsampling to simulate lower rate capture devices. Microphones are omnidirectional.

For the individual problems, please see  

1. [**Source Removal Details**](sourceremove.md)
  - Lab41 POC: Karl Ni
2. [**Moving Sources Details**](movingsource.md)
  - Lab41 POC: Patrick Callier & Abhinav Ganesh
3. [**Standoff Distance**](standoff.md)
  - Time Permitting

## Collaboration Efforts

- **Microphone Equipment**
  - Lab41 POC: Abhinav Ganesh
  - GELB Music (Redwood City) [Riley Bradley](mailto:riley@gelbmusic.com), 650-365-8878
  - [Notes from Meetings](micequipment.md)
- **Acoustic Modeling**
  - Lab41 POC: Patrick Callier
  - [Doug James](mailto:djames@stanford.edu) at Stanford.
  - [Notes from Meetings](acousticmodel.md)
- **Algorithm Help**
  - Lab41 POC: Karl Ni
  - [Arlo Faria](mailto:arlo@remeeting.com) at Re-Meeting.
  - [Notes from Meetings](algorithmhelp.md)

## Algorithms and Software

This project will be primarily an analysis of cost functions (in supervised and unsupervised settings) that can be used in order to denoise and isolate signals.

The following will be implemented from scratch.

- **Supervised Methods Cost Functions**
  - **Weeks 1 and 2 and 3**
    - Inverse problem
    - Gradient descent
    - Neural networks
    - Backpropagation
- **Unsupervised Cost Functions** 
  - **Weeks 4 and 5**
    - Independent Components Analysis and Beamforming
  - **Week 6 and 7** 
    - Deep ICA and Deep Reconstruction ICA
- **Recurrent Methods for Neural Networks**
  - **Week 8**
- **Joint space/time convolutions**
  - **Week 9**

