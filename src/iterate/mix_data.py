import json
import numpy as np
import pandas as pd
import h5py
from magnolia.features.preprocessing import undo_stft_features


def main():
    number_of_mixed_sample = 1
    rng_seed = 0
    output_file = "/data/fs4/home/jhetherly/Projects/Magnolia/data/data_partitions/test.csv"
    noise_metadata = pd.read_csv('/data/fs4/home/jhetherly/Projects/Magnolia/data/data_partitions/UrbanSound8K/time_volume_interference/training_set/in_sample_test.csv')
    speech_metadata = pd.read_csv('/data/fs4/home/jhetherly/Projects/Magnolia/data/data_partitions/LibriSpeech/main_split/in_sample_test.csv')
    noise_key_label = 'key'
    speech_key_label = 'key'
    noise_data = h5py.File('/data/fs4/home/jhetherly/datasets/UrbanSound8K/processed_audio.hdf5','r')
    speech_data = h5py.File('/data/fs4/home/jhetherly/datasets/LibriSpeech/processed_train-clean-100.hdf5','r')
    noise_preprocessing_parameters = json.load(open('/data/fs4/home/jhetherly/Projects/Magnolia/settings/preprocess_UrbanSound8K.json'))
    speech_preprocessing_parameters = json.load(open('/data/fs4/home/jhetherly/Projects/Magnolia/settings/preprocess_LibriSpeech.json'))

    rng = np.random.RandomState(rng_seed)
    number_of_noise_samples = len(noise_metadata.index)
    number_of_speech_samples = len(speech_metadata.index)
    noise_preprocessing_parameters = noise_preprocessing_parameters['processing_parameters']
    del noise_preprocessing_parameters['track']
    del noise_preprocessing_parameters['overlap']
    noise_preprocessing_parameters['sample_rate'] = noise_preprocessing_parameters['output_sample_rate']
    del noise_preprocessing_parameters['output_sample_rate']
    speech_preprocessing_parameters = speech_preprocessing_parameters['processing_parameters']
    del speech_preprocessing_parameters['track']
    del speech_preprocessing_parameters['overlap']
    speech_preprocessing_parameters['sample_rate'] = speech_preprocessing_parameters['output_sample_rate']
    del speech_preprocessing_parameters['output_sample_rate']

    for _ in range(number_of_mixed_sample):
        noise_index = rng.randint(number_of_noise_samples)
        speech_index = rng.randint(number_of_speech_samples)
        noise_key = noise_metadata.get_value(noise_index, noise_key_label)
        speech_key = speech_metadata.get_value(speech_index, speech_key_label)

        noise_spectrogram = noise_data[noise_key]
        speech_spectrogram = speech_data[speech_key]

        print(noise_spectrogram[:].shape)
        print(undo_stft_features(noise_spectrogram[:], **noise_preprocessing_parameters))
        print(undo_stft_features(speech_spectrogram[:], **speech_preprocessing_parameters))
        print(np.abs(noise_spectrogram))


if __name__ == '__main__':
    main()
