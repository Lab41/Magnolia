
import os.path
import argparse
import logging.config
import json
import numpy as np
import h5py
import librosa as lr
import msgpack
from magnolia.features.preprocessing import undo_preprocessing
from magnolia.iterate.mix_iterator import MixIterator
from magnolia.utils.postprocessing import convert_preprocessing_parameters


def apply_binary_mask(mask, stft):
    return np.abs(stft)*mask*np.exp(1j* np.angle(stft))


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Reconstruct waveforms from mixes.')
    parser.add_argument('--sample', '-n',
                        default=1,
                        type=int,
                        help='sample number to write to file (1-indexed)')
    parser.add_argument('--output_file', '-o',
                        default='mix.wav',
                        help='output file name (wav format)')
    parser.add_argument('--settings', '-s',
                        default='../../settings/mixing_template.json',
                        help='sample mixing settings JSON file')
    parser.add_argument('--logger_settings', '-l',
                        default='../../settings/logging.conf',
                        help='logging configuration file')
    args = parser.parse_args()

    # Load logging configuration
    logging.config.fileConfig(args.logger_settings)
    logger = logging.getLogger('iteration')

    mixer = MixIterator([args.settings], batch_size=1)
    mixer_iter = iter(mixer)

    with open(args.settings) as settings_file:
        settings = json.load(settings_file)
        total_number_of_mixed_samples = settings['number_of_mixed_samples']
        byte_buffer = settings['byte_buffer']
        samples_file = settings['output_file']

        assert(args.sample <= total_number_of_mixed_samples and args.sample > 0)

        signal = settings['signals'][0]
        preprocessing_settings = json.load(open(signal['preprocessing_settings']))
        istft_args = convert_preprocessing_parameters(preprocessing_settings['processing_parameters']['stft_args'])
        preemphasis_coeff = preprocessing_settings['processing_parameters']['preemphasis_coeff']
        sample_rate = preprocessing_settings['processing_parameters']['target_sample_rate']
        sample_length = settings['target_sample_length']
        total_length = int(sample_length*sample_rate)

        for i in range(args.sample):
            spec, bin_masks, uids, snrs = next(mixer_iter)

        spec = spec[0]
        bin_masks = bin_masks[0]
        uids = uids[0]
        snrs = snrs[0]

        print('SNR of this mix: {}'.format(snrs))

        mix_file_name = '{}_mix.wav'.format(os.path.splitext(args.output_file)[0])
        y = undo_preprocessing(spec, total_length,
            preemphasis_coeff=preemphasis_coeff,
            istft_args=istft_args)
        lr.output.write_wav(mix_file_name, y, sample_rate, norm=True)

        for i in range(bin_masks.shape[0]):
            source_file_name = '{}_{}.wav'.format(os.path.splitext(args.output_file)[0], uids[i])
            source_spec = apply_binary_mask(bin_masks[i], spec)
            source_y = undo_preprocessing(source_spec, total_length,
                preemphasis_coeff=preemphasis_coeff,
                istft_args=istft_args)
            lr.output.write_wav(source_file_name, source_y, sample_rate, norm=True)


if __name__ == '__main__':
    main()
