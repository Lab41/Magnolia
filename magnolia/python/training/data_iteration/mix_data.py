import argparse
import logging.config
import json
import numpy as np
import msgpack
import tqdm
from magnolia.utils.sample import Sample
from magnolia.utils.mixing import construct_mixed_sample


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline.')
    parser.add_argument('--settings', '-s',
                        default='../../settings/mixing_template.json',
                        help='sample mixing settings JSON file')
    parser.add_argument('--logger_settings', '-l',
                        default='../../settings/logging.conf',
                        help='logging configuration file')
    args = parser.parse_args()

    # Load logging configuration
    logging.config.fileConfig(args.logger_settings)
    logger = logging.getLogger('partitioning')

    with open(args.settings) as settings_file:
        settings = json.load(settings_file)

        logger.debug('settings {}'.format(settings))

        rng = np.random.RandomState(settings['rng_seed'])
        number_of_mixed_samples = settings['number_of_mixed_samples']
        byte_buffer = settings['byte_buffer']
        snr_range = settings['snr_range']
        output_file_name = settings['output_file']
        target_sample_length = settings['target_sample_length']
        signals = []
        noises = []

        for signal_setting in settings['signals']:
            signals.append(Sample(rng, signal_setting))
        for noise_setting in settings['noises']:
            noises.append(Sample(rng, noise_setting))

        ofile = open(output_file_name, 'bw+')
        for _ in tqdm.trange(number_of_mixed_samples):
            snr = rng.uniform(*snr_range)

            result = construct_mixed_sample(signals, noises, snr, target_sample_length)

            bresult = msgpack.packb(result)
            ofile.write(bytes(format(len(bresult), str(byte_buffer)), 'utf-8'))
            ofile.write(bresult)


if __name__ == '__main__':
    main()
