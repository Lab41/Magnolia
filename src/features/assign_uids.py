import os.path
import argparse
import logging.config
import json
import h5py
import tqdm
import pandas as pd


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline.')
    parser.add_argument('--settings', '-s',
                        default='../../settings/assign_uids_template.json',
                        help='assign uid settings JSON file')
    parser.add_argument('--logger_settings', '-l',
                        default='../../settings/logging.conf',
                        help='logging configuration file')
    args = parser.parse_args()

    # Load logging configuration
    logging.config.fileConfig(args.logger_settings)
    logger = logging.getLogger('preprocessing')

    with open(args.settings) as settings_file:
        settings = json.load(settings_file)

        logger.debug('settings {}'.format(settings))

        output_filename = settings['output_file']
        count = 0
        uid_dict = {}
        to_store_uid_dict = {
            'uid': [],
            'local_id': [],
            'dataset_type': [],
        }
        for preprocessing_setting in tqdm.tqdm(settings['preprocessing_settings']):
            metadata_key = preprocessing_setting['metadata_id']
            data_label = preprocessing_setting['data_label']
            preprocessing_settings = json.load(open(preprocessing_setting['settings']))
            spectrogram_file = preprocessing_settings['spectrogram_output_file']
            metadata_file = pd.read_csv(preprocessing_settings['metadata_output_file'])
            dataset_type = preprocessing_settings['dataset_type']

            logger.info('starting assigning global uids to the {} dataset'.format(dataset_type))

            for key in metadata_file[metadata_key].unique():
                uid_dict[key] = count
                to_store_uid_dict['uid'].append(uid_dict[key])
                to_store_uid_dict['local_id'].append(key)
                to_store_uid_dict['dataset_type'].append(dataset_type)
                count += 1

            f = h5py.File(spectrogram_file, 'a')
            total_number_of_rows = len(metadata_file.index)
            pbar = tqdm.tqdm(total=total_number_of_rows, leave=False)
            for index, row in metadata_file.iterrows():
                f[row[data_label]].attrs['uid'] = uid_dict[row[metadata_key]]
                pbar.update(1)
            pbar.close()

            output_filename = '{}_{}.csv'.format(os.path.splitext(output_filename)[0], dataset_type)

        pd.DataFrame.from_dict(to_store_uid_dict).to_csv(output_filename)


if __name__ == '__main__':
    main()
