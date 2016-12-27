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
2. [**Moving Sources Details**](movingsource.md)
3. [**Standoff Distance**](standoff.md)

## Collaboration Efforts

## Algorithms and Software

This project will be primarily an analysis of cost functions (in supervised and unsupervised settings) that can be used in order to denoise and isolate signals.

Independent Component Analysis
- **Week 1** - Gradient Descent and 


