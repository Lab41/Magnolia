import argparse
import logging.config
import json
import pandas as pd
import msgpack


def main():

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert binary mixes to a readable CSV format.')
    parser.add_argument('--settings', '-s',
                        default='../../settings/mixing_template.json',
                        help='mix settings JSON file')
    parser.add_argument('--logger_settings', '-l',
                        default='../../settings/logging.conf',
                        help='logging configuration file')
    args = parser.parse_args()

    # Load logging configuration
    logging.config.fileConfig(args.logger_settings)
    logger = logging.getLogger('iteration')

    mix_settings = json.load(open(args.settings))
    byte_buffer = int(mix_settings['byte_buffer'])
    samples_file = open(mix_settings['output_file'], 'br')

    new_mix_info = {'snr': [],
                    'snr_factor': []}
    while True:
        try:
            mix_info = msgpack.unpackb(samples_file.read(int(bytes.decode(samples_file.read(byte_buffer)))), encoding='utf-8')

            new_mix_info['snr'].append(mix_info['snr'])
            new_mix_info['snr_factor'].append(mix_info['snr_factor'])

            signal_keys = mix_info['signal_keys']
            for i in range(len(signal_keys)):
                signal_key_name = 'signal{}_keys'.format(i + 1)
                if signal_key_name not in new_mix_info:
                    new_mix_info[signal_key_name] = []
                    new_mix_info['signal{}_scale_factors'.format(i + 1)] = []
                    new_mix_info['signal{}_spectrogram_starts'.format(i + 1)] = []
                    new_mix_info['signal{}_spectrogram_ends'.format(i + 1)] = []
                new_mix_info[signal_key_name].append(mix_info['signal_keys'][i])
                new_mix_info['signal{}_scale_factors'.format(i + 1)].append(mix_info['signal_scale_factors'][i])
                new_mix_info['signal{}_spectrogram_starts'.format(i + 1)].append(mix_info['signal_spectrogram_starts'][i])
                new_mix_info['signal{}_spectrogram_ends'.format(i + 1)].append(mix_info['signal_spectrogram_ends'][i])

            noise_keys = mix_info['noise_keys']
            for i in range(len(noise_keys)):
                noise_key_name = 'noise{}_keys'.format(i + 1)
                if noise_key_name not in new_mix_info:
                    new_mix_info[noise_key_name] = []
                    new_mix_info['noise{}_scale_factors'.format(i + 1)] = []
                    new_mix_info['noise{}_spectrogram_starts'.format(i + 1)] = []
                    new_mix_info['noise{}_spectrogram_ends'.format(i + 1)] = []
                new_mix_info[noise_key_name].append(mix_info['noise_keys'][i])
                new_mix_info['noise{}_scale_factors'.format(i + 1)].append(mix_info['noise_scale_factors'][i])
                new_mix_info['noise{}_spectrogram_starts'.format(i + 1)].append(mix_info['noise_spectrogram_starts'][i])
                new_mix_info['noise{}_spectrogram_ends'.format(i + 1)].append(mix_info['noise_spectrogram_ends'][i])

        except ValueError:
            break

    pd.DataFrame(new_mix_info).to_csv(mix_settings['output_file'].replace('.bin', '.csv'))


if __name__ == '__main__':
    main()
