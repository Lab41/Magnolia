import logging.config
import json
import numpy as np
import h5py
import msgpack
from magnolia.utils.mixing import convert_sample_length_to_nframes


logger = logging.getLogger('iteration')


class MixIterator:
    def __init__(self, mixes_settings_filenames, batch_size=1, from_disk=True):
        """Initializes iterator given a list of mixes and desired batch size
        """
        self._batch_size = batch_size
        self._number_of_mix_sets = len(mixes_settings_filenames)
        self._current_mix_set = 0
        self._sample_dimensions = None
        self._batch = None
        self._mask_batch = None
        self._uid_batch = None
        self._snr_batch = None
        self._sample_rate = 0
        self._sample_length = 0
        self._number_of_samples_in_mixes = 0
        self._total_numbers_of_mixed_samples = []
        self._byte_buffers = []
        self._signal_spec_data = []
        self._noise_spec_data = []
        self._samples_files = []

        mixes_settings_json = []
        for mix_settings_filename in mixes_settings_filenames:
            mixes_settings_json.append(json.load(open(mix_settings_filename)))

        preemphasis_coeff = None
        sample_rate = None
        sample_length = None
        target_sample_lengths = []
        numbers_of_samples_in_mixes = []
        spec_file_dict = {}
        for settings in mixes_settings_json:
            self._total_numbers_of_mixed_samples.append(settings['number_of_mixed_samples'])
            self._byte_buffers.append(settings['byte_buffer'])
            self._samples_files.append(open(settings['output_file'], 'br'))
            target_sample_lengths.append(settings['target_sample_length'])
            numbers_of_samples_in_mixes.append(0)

            spec_data = []
            for i, signal in enumerate(settings['signals']):
                preprocessing_settings = json.load(open(signal['preprocessing_settings']))
                if preprocessing_settings['spectrogram_output_file'] not in spec_file_dict:
                    if from_disk:
                        spec_file_dict[preprocessing_settings['spectrogram_output_file']] = h5py.File(preprocessing_settings['spectrogram_output_file'], 'r')
                    else:
                        spec_file_dict[preprocessing_settings['spectrogram_output_file']] = h5py.File(preprocessing_settings['spectrogram_output_file'], 'r', driver='core')
                spec_data.append(spec_file_dict[preprocessing_settings['spectrogram_output_file']])
                numbers_of_samples_in_mixes[-1] += 1

                if self._sample_dimensions is None:
                    n_fft = 2048
                    stft_args = preprocessing_settings['processing_parameters']['stft_args']
                    if 'n_fft' in stft_args:
                        n_fft = stft_args['n_fft']
                    target_sample_rate = preprocessing_settings['processing_parameters']['target_sample_rate']
                    self._sample_length_in_bits = int(target_sample_rate*settings['target_sample_length'])
                    n_frames = convert_sample_length_to_nframes(self._sample_length_in_bits, **stft_args)
                    self._sample_dimensions = (1 + n_fft//2, n_frames)

                if preemphasis_coeff is None:
                    preemphasis_coeff = preprocessing_settings['processing_parameters']['preemphasis_coeff']
                else:
                    # TODO: throw proper exception
                    assert(preemphasis_coeff == preprocessing_settings['processing_parameters']['preemphasis_coeff'])
                if sample_rate is None:
                    sample_rate = preprocessing_settings['processing_parameters']['target_sample_rate']
                else:
                    # TODO: throw proper exception
                    assert(sample_rate == preprocessing_settings['processing_parameters']['target_sample_rate'])
                if sample_length is None:
                    sample_length = settings['target_sample_length']
                else:
                    # TODO: throw proper exception
                    assert(sample_length == settings['target_sample_length'])
            self._signal_spec_data.append(spec_data)

            spec_data = []
            for i, noise in enumerate(settings['noises']):
                preprocessing_settings = json.load(open(noise['preprocessing_settings']))
                if preprocessing_settings['spectrogram_output_file'] not in spec_file_dict:
                    if from_disk:
                        spec_file_dict[preprocessing_settings['spectrogram_output_file']] = h5py.File(preprocessing_settings['spectrogram_output_file'], 'r')
                    else:
                        spec_file_dict[preprocessing_settings['spectrogram_output_file']] = h5py.File(preprocessing_settings['spectrogram_output_file'], 'r', driver='core')
                spec_data.append(spec_file_dict[preprocessing_settings['spectrogram_output_file']])
                numbers_of_samples_in_mixes[-1] += 1

                if preemphasis_coeff is None:
                    preemphasis_coeff = preprocessing_settings['processing_parameters']['preemphasis_coeff']
                else:
                    # TODO: throw proper exception
                    assert(preemphasis_coeff == preprocessing_settings['processing_parameters']['preemphasis_coeff'])
                if sample_rate is None:
                    sample_rate = preprocessing_settings['processing_parameters']['target_sample_rate']
                else:
                    # TODO: throw proper exception
                    assert(sample_rate == preprocessing_settings['processing_parameters']['target_sample_rate'])
                if sample_length is None:
                    sample_length = settings['target_sample_length']
                else:
                    # TODO: throw proper exception
                    assert(sample_length == settings['target_sample_length'])
            self._noise_spec_data.append(spec_data)

        target_sample_length = target_sample_lengths[0]
        for tsl in target_sample_lengths:
            # TODO: throw proper exception
            assert(np.allclose([target_sample_length], [tsl]))
        self._sample_rate = sample_rate
        self._sample_length = sample_length
        self._number_of_samples_in_mixes = numbers_of_samples_in_mixes[0]
        for nsm in numbers_of_samples_in_mixes:
            # TODO: throw proper exception
            assert(np.allclose([self._number_of_samples_in_mixes], [nsm]))
        self._batch = np.zeros((self._batch_size, self._sample_dimensions[0], self._sample_dimensions[1]), dtype=np.complex128)
        self._mask_batch = np.zeros((self._batch_size, self._number_of_samples_in_mixes, self._sample_dimensions[0], self._sample_dimensions[1]), dtype=bool)
        self._source_batch = np.zeros((self._batch_size, self._number_of_samples_in_mixes, self._sample_dimensions[0], self._sample_dimensions[1]), dtype=np.complex128)
        self._uid_batch = np.zeros((self._batch_size, self._number_of_samples_in_mixes), dtype=int)
        self._snr_batch = np.zeros((self._batch_size), dtype=float)

    def __next__(self):
        sample_count = 1
        mix_info = None
        f = self._samples_files[self._current_mix_set]
        byte_buffer = self._byte_buffers[self._current_mix_set]
        while True:
            if sample_count > self._batch_size:
                break
            try:
                mix_info = msgpack.unpackb(f.read(int(bytes.decode(f.read(byte_buffer)))), encoding='utf-8')
                self._construct_sample_from_mix_info(mix_info, sample_count - 1)
                if sample_count >= self._batch_size:
                    break
            except ValueError:
                self._samples_files[self._current_mix_set].seek(0)
                self._current_mix_set += 1
                if self._current_mix_set == self._number_of_mix_sets:
                    self._current_mix_set = 0
                    raise StopIteration
                f = self._samples_files[self._current_mix_set]
                byte_buffer = self._byte_buffers[self._current_mix_set]
                continue
            sample_count += 1
        return self._batch, self._mask_batch, self._source_batch, self._uid_batch, self._snr_batch

    def __iter__(self):
        return self

    def sample_dimensions(self):
        return self._sample_dimensions

    def epoch_size(self):
        epoch_size = 0
        for number_of_mixed_samples in self._total_numbers_of_mixed_samples:
            epoch_size += number_of_mixed_samples
        return epoch_size

    def sample_rate(self):
        return self._sample_rate

    def sample_length(self):
        return self._sample_length

    def sample_length_in_bits(self):
        return self._sample_length_in_bits

    def number_of_samples_in_mixes(self):
        return self._number_of_samples_in_mixes

    def _construct_sample_from_mix_info(self, mix_info, batch_number):
        # total_spec = None
        assigned_spec = False
        specs = []
        snr_factor = mix_info['snr_factor']

        for i, data in enumerate(self._signal_spec_data[self._current_mix_set]):
            key = mix_info['signal_keys'][i]
            uid = data[key].attrs['uid']
            scale_factor = mix_info['signal_scale_factors'][i]
            spectrogram_start = mix_info['signal_spectrogram_starts'][i]
            spectrogram_end = mix_info['signal_spectrogram_ends'][i]
            spectrogram = scale_factor*data[key][:, spectrogram_start:spectrogram_end]

            specs.append(spectrogram)
            self._uid_batch[batch_number, i] = uid

            if not assigned_spec:
                self._batch[batch_number, :, :] = spectrogram
                self._snr_batch[batch_number] = mix_info['snr']
                assigned_spec = True
            else:
                self._batch[batch_number] += spectrogram
            # if total_spec is None:
            #     total_spec = spectrogram
            # else:
            #     total_spec += spectrogram

        signal_count_offset = len(self._signal_spec_data[self._current_mix_set])
        for i, data in enumerate(self._noise_spec_data[self._current_mix_set]):
            key = mix_info['noise_keys'][i]
            uid = data[key].attrs['uid']
            scale_factor = mix_info['noise_scale_factors'][i]
            spectrogram_start = mix_info['noise_spectrogram_starts'][i]
            spectrogram_end = mix_info['noise_spectrogram_ends'][i]
            spectrogram = (snr_factor*scale_factor)*data[key][:, spectrogram_start:spectrogram_end]

            specs.append(spectrogram)
            self._uid_batch[batch_number, i + signal_count_offset] = uid

            if not assigned_spec:
                self._batch[batch_number, :, :] = spectrogram
                self._snr_batch[batch_number] = mix_info['snr']
                assigned_spec = True
            else:
                self._batch[batch_number] += spectrogram
            # if total_spec is None:
            #     total_spec = spectrogram
            # else:
            #     total_spec += spectrogram

        for i, spec in enumerate(specs):
            self._source_batch[batch_number, i, :, :] = spec
            self._mask_batch[batch_number, i, :, :] = np.abs(spec) >= np.abs(self._batch[batch_number] - spec)
            # self._mask_batch[batch_number, i, :, :] = np.abs(spec) >= np.abs(total_spec - spec)

        # self._batch[batch_number, :, :] = np.abs(total_spec)

# kernprof -l mix_iterator.py
# python -m line_profiler mix_iterator.py.lprof

# NOTE: most of the time is spent reading from disk
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#    145                                               @profile
#    146                                               def construct_sample_from_mix_info(self, mix_info, batch_number):
#    147     10000         7922      0.8      0.0          total_spec = None
#    148     10000         7114      0.7      0.0          specs = []
#    149     10000        12742      1.3      0.0          snr_factor = mix_info['snr_factor']
#    150
#    151     20000        39781      2.0      0.0          for i, data in enumerate(self._signal_spec_data[self._current_mix_set]):
#    152     10000         8285      0.8      0.0              key = mix_info['signal_keys'][i]
#    153     10000      4883503    488.4      4.9              uid = data[key].attrs['uid']
#    154     10000        19098      1.9      0.0              scale_factor = mix_info['signal_scale_factors'][i]
#    155     10000         8679      0.9      0.0              spectrogram_start = mix_info['signal_spectrogram_starts'][i]
#    156     10000         7723      0.8      0.0              spectrogram_end = mix_info['signal_spectrogram_ends'][i]
#    157     10000     60384337   6038.4     60.7              spectrogram = scale_factor*data[key][:, spectrogram_start:spectrogram_end]
#    158
#    159     10000        33513      3.4      0.0              specs.append(spectrogram)
#    160     10000        38965      3.9      0.0              self._uid_batch[batch_number, i] = uid
#    161
#    162     10000         8526      0.9      0.0              if total_spec is None:
#    163     10000         7675      0.8      0.0                  total_spec = spectrogram
#    164                                                       else:
#    165                                                           total_spec += spectrogram
#    166
#    167     10000        21887      2.2      0.0          signal_count_offset = len(self._signal_spec_data[self._current_mix_set])
#    168     20000        54441      2.7      0.1          for i, data in enumerate(self._noise_spec_data[self._current_mix_set]):
#    169     10000        16119      1.6      0.0              key = mix_info['noise_keys'][i]
#    170     10000      3601043    360.1      3.6              uid = data[key].attrs['uid']
#    171     10000        22085      2.2      0.0              scale_factor = mix_info['noise_scale_factors'][i]
#    172     10000        10482      1.0      0.0              spectrogram_start = mix_info['noise_spectrogram_starts'][i]
#    173     10000         9849      1.0      0.0              spectrogram_end = mix_info['noise_spectrogram_ends'][i]
#    174     10000     21555187   2155.5     21.7              spectrogram = (snr_factor*scale_factor)*data[key][:, spectrogram_start:spectrogram_end]
#    175
#    176     10000        24081      2.4      0.0              specs.append(spectrogram)
#    177     10000        29525      3.0      0.0              self._uid_batch[batch_number, i + signal_count_offset] = uid
#    178
#    179     10000         8327      0.8      0.0              if total_spec is None:
#    180                                                           total_spec = spectrogram
#    181                                                       else:
#    182     10000       590740     59.1      0.6                  total_spec += spectrogram
#    183
#    184     30000        51255      1.7      0.1          for i, spec in enumerate(specs):
#    185     20000      6480734    324.0      6.5              self._mask_batch[batch_number, i, :, :] = np.abs(spec) >= np.abs(total_spec - spec)
#    186
#    187     10000      1615506    161.6      1.6          self._batch[batch_number, :, :] = np.abs(total_spec)


# if __name__ == '__main__':
#     import time
#     import argparse
#     # parse command line arguments
#     parser = argparse.ArgumentParser(description='Reconstruct waveforms from mixes.')
#     parser.add_argument('--settings', '-s',
#                         default='../../settings/mixing_template.json',
#                         help='sample mixing settings JSON file')
#     args = parser.parse_args()
#
#
#     mixer = MixIterator([args.settings], batch_size=256)
#     batch_count = 0
#     start = time.perf_counter()
#     for batch in iter(mixer):
#         batch_count += 1
#     end = time.perf_counter()
#     print('number of batches {}'.format(batch_count))
#     print('total time {}'.format(end - start))
#     print('average time per batch {}'.format((end - start)/batch_count))
