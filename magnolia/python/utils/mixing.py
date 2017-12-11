"""Functions to aid in mixing samples

TODO: need to add logging, documentation, and error handling
"""


import logging.config
import numpy as np
import librosa as lr


logger = logging.getLogger('partitioning')


def convert_sample_to_nframes(y_sample_start, y_sample_end, **stft_args):
    n_fft = 2048
    if 'n_fft' in stft_args:
        n_fft = stft_args['n_fft']
    hop_length = n_fft//4
    if 'hop_length' in stft_args:
        hop_length = stft_args['hop_length']
    return lr.samples_to_frames(np.array([y_sample_start, y_sample_end]),
                                hop_length=hop_length, n_fft=n_fft)


def convert_sample_length_to_nframes(y_sample_length, **stft_args):
    stft_frames = convert_sample_to_nframes(0, y_sample_length, **stft_args)
    return stft_frames[1] - stft_frames[0]


def compatable_preprocessing_parameters_for_mixing(params1, params2):
    default_nfft = 2048
    default_win_length = default_nfft
    default_hop_length = default_nfft//4

    result = params1['target_sample_rate'] == params2['target_sample_rate']
    if 'n_fft' in params1['stft_args'] and 'n_fft' in params2['stft_args']:
        result &= (params1['stft_args']['n_fft'] == params2['stft_args']['n_fft'])
    elif 'n_fft' not in params1['stft_args'] and 'n_fft' in params2['stft_args']:
        result &= (default_nfft == params2['stft_args']['n_fft'])
    elif 'n_fft' in params1['stft_args'] and 'n_fft' not in params2['stft_args']:
        result &= (params1['stft_args']['n_fft'] == default_nfft)
    if 'win_length' in params1['stft_args'] and 'win_length' in params2['stft_args']:
        result &= (params1['stft_args']['win_length'] == params2['stft_args']['win_length'])
    elif 'win_length' not in params1['stft_args'] and 'win_length' in params2['stft_args']:
        result &= (default_win_length == params2['stft_args']['win_length'])
    elif 'win_length' in params1['stft_args'] and 'win_length' not in params2['stft_args']:
        result &= (params1['stft_args']['win_length'] == default_win_length)
    if 'hop_length' in params1['stft_args'] and 'hop_length' in params2['stft_args']:
        result &= (params1['stft_args']['hop_length'] == params2['stft_args']['hop_length'])
    elif 'hop_length' not in params1['stft_args'] and 'hop_length' in params2['stft_args']:
        result &= (default_hop_length == params2['stft_args']['hop_length'])
    elif 'hop_length' in params1['stft_args'] and 'hop_length' not in params2['stft_args']:
        result &= (params1['stft_args']['hop_length'] == default_hop_length)
    return result


def compute_waveform_snr_factor(dB):
    # https://en.wikipedia.org/wiki/Signal-to-noise_ratio#Definition
    # ratio of variances since both signal and noise are assumed zero-meaned
    return np.power(10., -dB/20.)


def construct_mixed_sample(signals, noises, snr, target_sample_length):
    signal_scale_factors = np.ones(len(signals))
    noise_scale_factors = np.ones(len(noises))
    snr_noise_scale_factor = 1.0
    total_signal_sample_y = None
    total_noise_sample_y = None
    signal_samples = []
    noise_samples = []
    signal_preprocessing_parameters = None
    noise_preprocessing_parameters = None

    for i, signal in enumerate(signals):
        signal_samples.append(signal.draw_sample(target_sample_length))
        sample_y = signal_samples[-1]['sample_y']
        factor = compute_waveform_snr_factor(signal.dB_scale)
        signal_scale_factors[i] = factor
        if total_signal_sample_y is None:
            total_signal_sample_y = factor*sample_y/sample_y.std()
        else:
            total_signal_sample_y += factor*sample_y/sample_y.std()
        if signal_preprocessing_parameters is None:
            signal_preprocessing_parameters = signal.preprocessing_parameters
        elif compatable_preprocessing_parameters_for_mixing(signal_preprocessing_parameters, signal.preprocessing_parameters):
            # TODO: throw error
            assert(signal_preprocessing_parameters == signal.preprocessing_parameters)
    for i, noise in enumerate(noises):
        noise_samples.append(noise.draw_sample(target_sample_length))
        sample_y = noise_samples[-1]['sample_y']
        factor = compute_waveform_snr_factor(noise.dB_scale)
        noise_scale_factors[i] = factor
        if total_noise_sample_y is None:
            total_noise_sample_y = factor*sample_y/sample_y.std()
        else:
            total_noise_sample_y += factor*sample_y/sample_y.std()
        if noise_preprocessing_parameters is None:
            noise_preprocessing_parameters = noise.preprocessing_parameters
        elif compatable_preprocessing_parameters_for_mixing(noise_preprocessing_parameters, noise.preprocessing_parameters):
            # TODO: throw error
            assert(noise_preprocessing_parameters == noise.preprocessing_parameters)
    if compatable_preprocessing_parameters_for_mixing(signal_preprocessing_parameters, noise_preprocessing_parameters):
        # TODO: throw error
        assert(signal_preprocessing_parameters == noise_preprocessing_parameters)


    dB_factor = compute_waveform_snr_factor(snr)
    snr_noise_scale_factor = total_signal_sample_y.std()*dB_factor/total_noise_sample_y.std()

    result = {}
    result['snr'] = snr
    result['snr_factor'] = snr_noise_scale_factor
    result['signal_keys'] = []
    result['signal_scale_factors'] = []
    result['signal_spectrogram_starts'] = []
    result['signal_spectrogram_ends'] = []
    result['noise_keys'] = []
    result['noise_scale_factors'] = []
    result['noise_spectrogram_starts'] = []
    result['noise_spectrogram_ends'] = []

    for i, signal_sample in enumerate(signal_samples):
        result['signal_keys'].append(signal_sample['key'])
        result['signal_spectrogram_starts'].append(signal_sample['spectrogram_start'])
        result['signal_spectrogram_ends'].append(signal_sample['spectrogram_end'])
        result['signal_scale_factors'].append(np.asscalar(signal_scale_factors[i]))
    for i, noise_sample in enumerate(noise_samples):
        result['noise_keys'].append(noise_sample['key'])
        result['noise_spectrogram_starts'].append(noise_sample['spectrogram_start'])
        result['noise_spectrogram_ends'].append(noise_sample['spectrogram_end'])
        result['noise_scale_factors'].append(np.asscalar(noise_scale_factors[i]))

    return result
