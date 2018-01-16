import os
import json
import numpy as np
# import h5py
import pandas as pd
# import msgpack
import tqdm


def make_nice_table(input_data_file, mix_metadata_file, output_csv_file):
    print('processing input table {}\nwith metadata mix info {}'.format(input_data_file, mix_metadata_file))
    input_df = pd.read_csv(input_data_file, index_col='mix_number')
    mix_metadata_df = pd.read_csv(mix_metadata_file)

    result = {}
    result['Input_SNR'] = []
    result['Input_SDR'] = []
    result['Speaker_ID'] = []
    result['Speaker_Sex'] = []
    result['Noise_Type'] = []
    result['Output_SDR'] = []

    for mix_number in tqdm.tqdm(input_df.index.values):
        snr = input_df.loc[mix_number]['snr']
        initial_sdr = input_df.loc[mix_number]['source_1_original_sdr']
        speaker_id = mix_metadata_df.iloc[mix_number - 1]['signal1_keys'].split(os.path.sep)[1]
        speaker_sex = mix_metadata_df.iloc[mix_number - 1]['signal1_keys'].split(os.path.sep)[0]
        noise_type = mix_metadata_df.iloc[mix_number - 1]['noise1_keys'].split(os.path.sep)[1]
        output_sdr = input_df.loc[mix_number]['separated_source_1_output_sdr']

        result['Input_SNR'].append(snr)
        result['Input_SDR'].append(initial_sdr)
        result['Speaker_ID'].append(speaker_id)
        result['Speaker_Sex'].append(speaker_sex)
        result['Noise_Type'].append(noise_type)
        result['Output_SDR'].append(output_sdr)

    print('writing output CSV file to {}'.format(output_csv_file))
    pd.DataFrame(result).to_csv(output_csv_file, index=False)


def main():
    args = [
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/lab41/in_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/in_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/lab41/in_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/lab41/out_of_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/out_of_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/lab41/out_of_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/large_lab41/in_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/in_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/large_lab41/in_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/large_lab41/out_of_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/out_of_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/large_lab41/out_of_sample_test_sdr_summary.csv'],
        
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/mi_in_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/in_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/mi_in_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/mi_out_of_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/out_of_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/mi_out_of_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/dc_in_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/in_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/dc_in_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/dc_out_of_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/out_of_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/mask_sce/dc_out_of_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/mi_in_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/in_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/mi_in_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/mi_out_of_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/out_of_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/mi_out_of_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/dc_in_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/in_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/dc_in_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/dc_out_of_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/out_of_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/chimera/dc_out_of_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/snmf/in_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/in_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/snmf/in_sample_test_sdr_summary.csv'],
        ['/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/snmf/out_of_sample_test.csv',
         '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/aux/out_of_sample_test_mixes.csv',
         '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/evaluations/bss/snmf/out_of_sample_test_sdr_summary.csv']
    ]
    
    #args = args[:2]
    # args = args[2:4]
    #args = args[4:6]
    #args = args[6:8]
    #args = args[8:]
    for arg in args:
        make_nice_table(*arg)


if __name__ == '__main__':
    main()
