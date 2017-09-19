
"""Class for handling samples while mixing

TODO: need to add logging, documentation, and error handling
"""


from functools import lru_cache
import logging.config
import json
import numpy as np
import pandas as pd
import h5py
from magnolia.features.preprocessing import undo_preemphasis
from magnolia.utils.partition_graph import build_partition_graph, get_group_path
from magnolia.utils.mixing import convert_sample_to_nframes, convert_sample_length_to_nframes


logger = logging.getLogger('partitioning')


class Sample:
    def __init__(self, rng, settings):
        partition_settings = json.load(open(settings['partition_settings']))
        partition_graph_setting = json.load(open(partition_settings['partition_graphs_file']))['partition_graphs'][settings['partition_graph_number'] - 1]
        partition_graph_root = build_partition_graph(partition_settings['output_directory'], partition_graph_setting)
        preprocessing_settings = json.load(open(settings['preprocessing_settings']))
        self.preprocessing_parameters = preprocessing_settings['processing_parameters']
        self.partition_metadata = pd.read_csv(get_group_path(settings['group_name'], partition_graph_root))
        self.number_of_samples = len(self.partition_metadata.index)
        self.key = partition_graph_setting['data_label']
        self.wf_data = h5py.File(preprocessing_settings['waveform_output_file'], 'r')
        self.preemphasis_coeff = preprocessing_settings['processing_parameters']['preemphasis_coeff']
        self.sample_rate = preprocessing_settings['processing_parameters']['target_sample_rate']
        self.stft_parameters = preprocessing_settings['processing_parameters']['stft_args']
        self.dB_scale = 0.0 if 'dB_scale' not in settings else settings['dB_scale']
        self.rng = rng

        # these two are just function wrappers to make later calls more clear
        def _starting_frame(y_starting_point):
            stft_frames = convert_sample_to_nframes(y_starting_point,
                                                    y_starting_point,
                                                    **self.stft_parameters)
            return 0 if np.asscalar(stft_frames[0]) < 0 else np.asscalar(stft_frames[0])
        self.find_starting_frame = _starting_frame

        @lru_cache(maxsize=32)
        def _convert_sample_length_to_nframes(y_length):
            return np.asscalar(convert_sample_length_to_nframes(y_length, **self.stft_parameters))
        self.convert_sample_length_to_nframes = _convert_sample_length_to_nframes

    def draw_sample(self, target_sample_length):
        index = self.rng.randint(self.number_of_samples)
        key = self.partition_metadata.get_value(index, self.key)
        y = self.wf_data[key]
        y = undo_preemphasis(y, self.preemphasis_coeff)
        max_frames = self.wf_data[key].attrs['spectral_length']

        sample_y, sample_starting_point = self._sample_waveform(y,
                                        self.sample_rate, target_sample_length)
        starting_frame = self.find_starting_frame(sample_starting_point)
        frame_length = self.convert_sample_length_to_nframes(sample_y.size)
        if starting_frame + frame_length > max_frames:
            starting_frame = int(max_frames - frame_length - 1)

        return {
            "key": key,
            "y": y,
            "sample_y": sample_y[:],
            "sample_starting_point": sample_starting_point,
            "spectrogram_start": starting_frame,
            "spectrogram_end": starting_frame + frame_length
            }

    def _sample_waveform(self, y, sr, target_duration):
        """assumes sr is in bps and target_duration is in seconds"""
        number_of_samples = y.size
        number_of_target_waveform_samples = int(np.rint(sr*target_duration))
        if number_of_samples < number_of_target_waveform_samples:
            y = np.tile(y, int(np.ceil(number_of_target_waveform_samples/number_of_samples)))
            return y[:number_of_target_waveform_samples], 0
        valid_starting_points = number_of_samples - number_of_target_waveform_samples
        if valid_starting_points == 0:
            return y, 0
        starting_point = self.rng.randint(valid_starting_points)
        sliced_y = y[starting_point:starting_point + number_of_target_waveform_samples]
        return sliced_y, starting_point
