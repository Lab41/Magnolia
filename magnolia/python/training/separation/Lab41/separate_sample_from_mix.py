# Generic imports
import os
import json
import numpy as np
import pandas as pd
import librosa as lr

# Import Lab41's separation model
from magnolia.dnnseparate.L41model import L41Model

# Import utilities for using the model
from magnolia.utils.postprocessing import convert_preprocessing_parameters
from magnolia.features.preprocessing import undo_preprocessing
from magnolia.iterate.mix_iterator import MixIterator
from magnolia.utils.clustering_utils import l41_clustering_separate


def standardize_waveform(y):
    return (y - y.mean())/y.std()


def main():
    # from model settings
    model_params = {
        'nonlinearity': 'tanh',
        'layer_size': 600,
        'embedding_size': 40,
        'normalize': 'False'
    }
    uid_settings = '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/settings/assign_uids_LibriSpeech_UrbanSound8K.json'
    model_save_base = '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/model_saves/large_l41'

    model_location = '/cpu:0'
    model_settings = ''
    mixes = ['/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/settings/mixing_LibriSpeech_UrbanSound8K_test_in_sample.json']
    from_disk = True
    mix_number = 1
    output_path = '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/sample_wav_files/large_lab41'


    os.makedirs(output_path, exist_ok=True)

    mixer = MixIterator(mixes_settings_filenames=mixes,
                        batch_size=1,
                        from_disk=from_disk)

    # get frequency dimension
    frequency_dim = mixer.sample_dimensions()[0]

    # get number of sources
    settings = json.load(open(uid_settings))
    uid_file = settings['output_file']
    uid_csv = pd.read_csv(uid_file)
    number_of_sources = uid_csv['uid'].max() + 1

    model = L41Model(**model_params,
                     num_speakers=number_of_sources,
                     F=frequency_dim,
                     device=model_location)

    model.load(model_save_base)

    assert(mix_number <= mixer.epoch_size())

    mix_settings = json.load(open(mixes[0]))

    signal = mix_settings['signals'][0]
    preprocessing_settings = json.load(open(signal['preprocessing_settings']))
    stft_args = preprocessing_settings['processing_parameters']['stft_args']
    istft_args = convert_preprocessing_parameters(stft_args)
    preemphasis_coeff = preprocessing_settings['processing_parameters']['preemphasis_coeff']
    n_fft = 2048
    if 'n_fft' in stft_args:
        n_fft = stft_args['n_fft']


    for i in range(mix_number):
        spec, bin_masks, source_specs, uids, snrs = next(mixer)

    model_spec = spec
    spec = spec[0]
    bin_masks = bin_masks[0]
    source_specs = source_specs[0]
    uids = uids[0]
    snrs = snrs[0]

    print('SNR of this mix: {}'.format(snrs))

    y_mix = undo_preprocessing(spec, mixer.sample_length_in_bits(),
                               preemphasis_coeff=preemphasis_coeff,
                               istft_args=istft_args)

    # NOTE: this is only to make comparisons to the reconstructed waveforms later
    y_mix[-n_fft:] = 0.0
    y_mix = standardize_waveform(y_mix)

    # print('Mixed sample')
    lr.output.write_wav(os.path.join(output_path, 'mix_{}.wav'.format(mix_number)), y_mix, mixer.sample_rate(), norm=True)

    for i, source_spec in enumerate(source_specs):
        y = undo_preprocessing(source_spec, mixer.sample_length_in_bits(),
                               preemphasis_coeff=preemphasis_coeff,
                               istft_args=istft_args)

        # NOTE: this is only to make comparisons to the reconstructed waveforms later
        y[-n_fft:] = 0.0
        y = standardize_waveform(y)

        # print('Sample for source {}'.format(i + 1))
        lr.output.write_wav(os.path.join(output_path, 'mix_{}_original_source_{}.wav'.format(mix_number, i + 1)), y, mixer.sample_rate(), norm=True)

    source_specs = l41_clustering_separate(model_spec, model, mixer.number_of_samples_in_mixes())#, binary_mask=False)

    for i, source_spec in enumerate(source_specs):
        y = undo_preprocessing(source_spec, mixer.sample_length_in_bits(),
                               preemphasis_coeff=preemphasis_coeff,
                               istft_args=istft_args)

        # NOTE: this is only because the masking creates a chirp in the last
        #       fft frame (likely due to the mask)
        y[-n_fft:] = 0.0
        y = standardize_waveform(y)

        # print('Separated sample for source {}'.format(i + 1))
        lr.output.write_wav(os.path.join(output_path, 'mix_{}_separated_{}.wav'.format(mix_number, i + 1)), y, mixer.sample_rate(), norm=True)


if __name__ == '__main__':
    main()
