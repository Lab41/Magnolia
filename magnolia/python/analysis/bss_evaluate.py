import os
import glob
import re
import multiprocessing as mp
import numpy as np
import scipy.io.wavfile
import pandas as pd
import tqdm

from magnolia.utils.bss_eval import bss_eval_sources


def evaluate(input_path, output_csv_file, target_stype=None, eval_sr=8000, num_sources=2):
    print('starting evaluation on directory {}'.format(input_path))
    
    mix_glob_pattern = 'mix_*_snr_*.wav'
    mix_regex = r"mix_(?P<mix_number>[0-9]+)_snr_(?P<snr>[0-9\-\.]+).wav"
    original_source_glob_format = 'mix_{}_original_source_*.wav'
    original_regex = r"mix_(?P<mix_number>[0-9]+)_original_source_(?P<source_number>[0-9]+).wav"
    separated_source_glob_format = 'mix_{}_separated_source_*.wav'
    extended_separated_source_glob_format = 'mix_{}_*_separated_source_*.wav'
    separated_regex = r"mix_(?P<mix_number>[0-9]+)_separated_source_(?P<source_number>[0-9]+).wav"
    extended_separated_regex = r"mix_(?P<mix_number>[0-9]+)_(?P<stype>[a-zA-Z]*)_separated_source_(?P<source_number>[0-9]+).wav"

    mix_info = {'snr': [],
                'mix_number': [],
                'mix_file_location': []}
    for source_num in range(num_sources):
        mix_info['original_source_{}_file_location'.format(source_num + 1)] = []
        mix_info['separated_source_{}_file_location'.format(source_num + 1)] = []
        mix_info['source_{}_original_sdr'.format(source_num + 1)] = []
        mix_info['separated_source_{}_output_sdr'.format(source_num + 1)] = []

    mixes_list = glob.glob(os.path.join(input_path, mix_glob_pattern))
    for filename in tqdm.tqdm(mixes_list):
        dirname = os.path.dirname(os.path.normpath(filename))
        basename = os.path.basename(os.path.normpath(filename))
        m = re.match(mix_regex, basename)
        mix_num = int(m.group('mix_number'))
        mix_info['mix_number'].append(mix_num)
        mix_info['snr'].append(float(m.group('snr')))
        mix_info['mix_file_location'].append(filename)

        mix_input = []
        mix_y = scipy.io.wavfile.read(filename)[1]
        for i in range(num_sources):
            mix_input.append(mix_y)
        mix_input = np.stack(mix_input)

        original_input = []
        original_input_order = []
        original_source_glob = original_source_glob_format.format(mix_num)
        for original_filename in glob.glob(os.path.join(input_path, original_source_glob)):
            original_basename = os.path.basename(os.path.normpath(original_filename))
            m = re.match(original_regex, original_basename)
            source_num = int(m.group('source_number'))
            original_input_order.append(source_num)
            mix_info['original_source_{}_file_location'.format(source_num)].append(original_filename)
            original_y = scipy.io.wavfile.read(original_filename)[1]
            original_input.append(original_y)

        original_input = np.stack(original_input)[np.argsort(original_input_order)]

        separated_input = []
        separated_input_order = []
        separated_source_glob = separated_source_glob_format.format(mix_num)
        extended_separated_source_glob = extended_separated_source_glob_format.format(mix_num)
        is_extended = False
        gg = None
        if glob.glob(os.path.join(input_path, separated_source_glob)):
            gg = glob.glob(os.path.join(input_path, separated_source_glob))
        elif glob.glob(os.path.join(input_path, extended_separated_source_glob)):
            gg = glob.glob(os.path.join(input_path, extended_separated_source_glob))
            is_extended = True
        for separated_filename in gg:
            separated_basename = os.path.basename(os.path.normpath(separated_filename))
            m = None
            if not is_extended:
                m = re.match(separated_regex, separated_basename)
            else:
                m = re.match(extended_separated_regex, separated_basename)
                if m.group('stype') != target_stype:
                    continue
            source_num = int(m.group('source_number'))
            separated_input_order.append(source_num)
            mix_info['separated_source_{}_file_location'.format(source_num)].append(separated_filename)
            separated_y = scipy.io.wavfile.read(separated_filename)[1]
            separated_input.append(separated_y)

        separated_input = np.stack(separated_input)[np.argsort(separated_input_order)]

        starting_sdr, starting_sir, starting_sar, starting_perm = bss_eval_sources(original_input, mix_input)
        final_sdr, final_sir, final_sar, final_perm = bss_eval_sources(original_input, separated_input)

        for i in range(num_sources):
            mix_info['source_{}_original_sdr'.format(i + 1)].append(starting_sdr[i])
            mix_info['separated_source_{}_output_sdr'.format(i + 1)].append(final_sdr[final_perm][i])

    print('writing output CSV file to {}'.format(output_csv_file))
    pd.DataFrame(mix_info).to_csv(output_csv_file, index=False, index_label='mix_number')



if __name__ == '__main__':
    args = [
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/lab41/in_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/lab41/in_sample_test.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/lab41/out_of_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/lab41/out_of_sample_test.csv'],
        
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/large_lab41/in_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/large_lab41/in_sample_test.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/large_lab41/out_of_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/large_lab41/out_of_sample_test.csv'],
        
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/chimera/in_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/mi_in_sample_test.csv',
         'mi'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/chimera/out_of_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/mi_out_of_sample_test.csv',
         'mi'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/chimera/in_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/dc_in_sample_test.csv',
         'dc'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/chimera/out_of_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/dc_out_of_sample_test.csv',
         'dc'],
        
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/mask_sce/in_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/mi_in_sample_test.csv',
         'mi'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/mask_sce/out_of_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/mi_out_of_sample_test.csv',
         'mi'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/mask_sce/in_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/dc_in_sample_test.csv',
         'dc'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/mask_sce/out_of_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/dc_out_of_sample_test.csv',
         'dc'],
        
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/snmf/in_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/snmf/in_sample_test.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/snmf/out_of_sample_test',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/snmf/out_of_sample_test.csv']
    ]
    
    args = args[8:12]
    
    # Parallel
    #processes = []
    #for arg in args:
    #    processes.append(mp.Process(target=evaluate, args=arg))
    #    processes[-1].start()
    #    
    #for process in processes:
    #    process.join()
    
    # Parallel
    #pool = mp.Pool(processes=min(len(args), os.cpu_count() - 1))
    #pool = mp.Pool(processes=2)
    #pool.starmap(evaluate, args)
    
    # Sequential
    for arg in args:
        evaluate(*arg)
