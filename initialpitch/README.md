# uncal-beamform

## Necessary Python Libraries
### libsndfile
```brew install libsndfile```
### pyaudio
```
pip install scikits.audiolab
pip install pyaudio 
```

## Usage
### Record source sounds with

This will record wave files:
```
python recordwav.py <output-filename>
```
In order to use the above command with the beamforming code, you must have a set of wave files of the same prefix followed by the recording number used. For example, my filenames could be called

```
recording1.wav 
recording2.wav 
recording3.wav 
recording4.wav 
```

The prefix in this case would be `recording`. In the end, each of these four recordings denote a speaker at a specific location. 

Note that these are *not* the signals heard at the microphone. Those will be the sum of these recordings, each of which are delayed by a slight amount.

### Playback beamforming with
```
beamform.py <input-prefix> <speaker-id>
```
The speaker-id is an integer specifying which speaker you would like to hear. For example, in the above set of recordings, where the prefix is `recording`, if I'd like to hear a speaker at recording 3, the <speaker-id> would then be `3`.

For example, here is an example of what will work out of the box:

```
beamform.py wavs/labmates/recording- 16
```
