# Preprocessing Pipeline

This folder houses the code needed to run the data preprocessing pipeline.
This pipeline is responsible for converting the raw waveforms to short-time
Fourier transformed (stft) spectrograms and organizing these spectrograms in a
sensible manner for later iteration.
It's currently capable of running over the LibriSpeech and UrbanSound8K
datasets.
However, it's possible to run over other datasets with only a few alterations to
the pipeline script and settings file.

## Usage

The `preprocess_data.py` script is used to run the preprocessing pipeline.
This script takes as arguments a preprocessing settings file, logging
configuration file, and logger name.
Argument names can be displayed by running the script with the `--help` or `-h`
flag.
The most pertinent argument is the preprocessing settings JSON file; a template
of which can be found in `magnolia/settings/preprocessing_template.json`.
The structure of the JSON file is as follows:

```javascript
{
  "data_directory": "...", // path to "top-level" directory where data resides
  "metadata_file": "...", // file containing the metadata regarding the dataset (discussed later)
  "dataset_type": "...", // unique identifying string for this dataset (discussed later)
  "output_file": "...", // output HDF5 file name (with file extension)
  "file_type": "...", // input file extension (i.e. .wav)
  "processing_parameters": { // parameters mostly focused on the stft transform
    "output_sample_rate": 10000,
    "window_size": 0.0512,
    "overlap": 0.0256,
    "preemphasis_coeff": 0.95,
    "fft_size": 512,
    "track": null
  }
}
```

For a complete description of the `processing_parameters`, see the definition of
the `make_stft_dataset` function in `preprocessing.py`.

The output HDF5 file should contain the stft spectrograms structured in such a
way that it's convenient for iteration in later steps (i.e. training, etc.).

### Customization for new datasets

Most of the alterations needed for a new dataset are made to the preprocessing
settings JSON file.
However, one important change needs to be made to the `preprocess_data.py`
script.
If the output file is to have any hierarchical structure, a new "key maker"
class must be created.
The purpose of this class is to create an HDF5 key given the dataset metadata
and the file name of an input file.
The class only needs two methods: the `__init__` method which takes as it's only
argument the metadata file name (`metadata_file` from the setting file) and the
`__call__` that return a key given an input file name.
The name of the class is also important.
It's name should be formatted as `dataset_type` followed by `_key_maker`.
For instance, if the `dataset_type` is LibriSpeech, then the key maker class
for this dataset should be `LibriSpeech_key_maker`.
In summary, a class such as this should go at the top of the
`preprocess_data.py` script:

```python
class <dataset_type>_key_maker:
    """Creates keys for <dataset> given its metadata and a filename"""
    def __init__(self, metadata_path):
        # do something with metadata here (such as storing it as a DataFrame)
        pass

    def __call__(self, filename):
        key = ''
        # do something with filename and metadata
        return key
```

If the "key maker" class is omitted, then the resulting HDF5 file will lack any
grouping information.

## Feature Extraction

This folder gives examples of using spectrograms and mfcc's, in order to do the necessary processing.

## Spectrogram Features
The basis of many of the algorithms is the short time Fourier transform (STFT). These functions have been implemented.

## Mel Frequency Cepstral Coefficients
These are defined as:

$$\mathbf{C} = \mathcal{DCT} \left\{ \Phi \log |\mathcal{DFT}(\hat\mathbf{x})|   \right\} $$

![Mel Frequencies](images/melfreq.png)
