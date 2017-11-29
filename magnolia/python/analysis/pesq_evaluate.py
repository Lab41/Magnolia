import os
import glob
import re
import subprocess
import pandas as pd


def main():
    output_path = '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/in_sample_test'
    output_csv_file = '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/in_sample_test.csv'
    pesq_executable = '/data/fs4/home/jhetherly/src/ITU-T_pesq/bin/itu-t-pesq2005'
    eval_sr = 8000
    num_sources = 2

    mix_glob_pattern = 'mix_*_snr_*.wav'
    mix_regex = r"mix_(?P<mix_number>[0-9]+)_snr_(?P<snr>[0-9\-\.]+).wav"
    original_source_glob_format = 'mix_{}_original_source_*.wav'
    original_regex = r"mix_(?P<mix_number>[0-9]+)_original_source_(?P<source_number>[0-9]+).wav"
    separated_source_glob_format = 'mix_{}_separated_source_*.wav'
    separated_regex = r"mix_(?P<mix_number>[0-9]+)_separated_source_(?P<source_number>[0-9]+).wav"

    mix_info = {'snr': [],
                'mix_number': [],
                'mix_file_location': []}
    for source_num in range(num_sources):
        mix_info['original_source_{}_file_location'.format(source_num + 1)] = []
        mix_info['separated_source_{}_file_location'.format(source_num + 1)] = []
    for filename in glob.glob(os.path.join(output_path, mix_glob_pattern)):
        dirname = os.path.dirname(os.path.normpath(filename))
        basename = os.path.basename(os.path.normpath(filename))
        m = re.match(mix_regex, basename)
        mix_num = int(m.group('mix_number'))
        mix_info['mix_number'].append(mix_num)
        mix_info['snr'].append(float(m.group('snr')))
        mix_info['mix_file_location'].append(filename)

        original_filenames = []
        original_source_glob = original_source_glob_format.format(mix_num)
        for original_filename in glob.glob(os.path.join(output_path, original_source_glob)):
            original_basename = os.path.basename(os.path.normpath(original_filename))
            m = re.match(original_regex, original_basename)
            source_num = int(m.group('source_number'))
            mix_info['original_source_{}_file_location'.format(source_num)].append(original_filename)
            original_filenames.append(original_filename)

            # run the pesq metric calculator for each original source against mix
            command = [pesq_executable,
                       original_filename, filename,
                       "+{}".format(eval_sr)]
            print("Mix: {}\nRunning: {}".format(mix_num, ' '.join(command)))
            subprocess.run(command, stdout=subprocess.DEVNULL)

        separated_source_glob = separated_source_glob_format.format(mix_num)
        for separated_filename in glob.glob(os.path.join(output_path, separated_source_glob)):
            separated_basename = os.path.basename(os.path.normpath(separated_filename))
            m = re.match(separated_regex, separated_basename)
            source_num = int(m.group('source_number'))
            mix_info['separated_source_{}_file_location'.format(source_num)].append(separated_filename)

            # run the pesq metric calculator for each original source against separated sources
            for original_filename in original_filenames:
                command = [pesq_executable,
                           original_filename, separated_filename,
                           "+{}".format(eval_sr)]
                print("Mix: {}\nRunning: {}".format(mix_num, ' '.join(command)))
                subprocess.run(command, stdout=subprocess.DEVNULL)

    pd.DataFrame(mix_info).to_csv(output_csv_file)


if __name__ == '__main__':
    main()
