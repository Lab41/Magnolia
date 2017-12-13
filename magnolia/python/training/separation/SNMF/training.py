# Generic imports
import json
import numpy as np
import pandas as pd

# Import sparse nmf function
from magnolia.factorization.snmf import SNMF

from magnolia.iterate.mix_iterator import MixIterator


def main():

    params = {}
    params['cf'] = 'kl'
    params['sparsity'] = 5
    params['R'] = 1000
    params['max_iter'] = 100
    params['conv_eps'] = 1e-4
    params['verbose'] = True
    T_L = 8
    T_R = 0
    random_seed = 1234567890
    uid_settings = '/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/settings/assign_uids_LibriSpeech_UrbanSound8K.json'
    library_output_file = '/local_data/magnolia/experiment_data/date_2017_09_28_time_13_14/aux/model_saves/snmf/library_weights.hdf5'
    mixes = ['/local_data/magnolia/pipeline_data/date_2017_09_27_time_13_25/settings/mixing_LibriSpeech_UrbanSound8K_train.json']
    from_disk = True
    batch_size = 5000
    num_batches = 1
    read_checkpoint = False
    library_checkpoint_file = ''

    model = SNMF(T_L, T_R, params['R'], params['sparsity'], params['cf'])

    mixer = MixIterator(mixes_settings_filenames=mixes,
                        batch_size=1,
                        from_disk=from_disk)

    # get frequency dimension
    frequency_dim = mixer.sample_dimensions()[0]

    # get number of sources
    settings = json.load(open(uid_settings))
    uid_file = settings['output_file']
    uid_csv = pd.read_csv(uid_file)
    number_of_sources = uid_csv['uid'].max() + 1

    settings = json.load(open(mixes[0]))

    signal = settings['signals'][0]
    preprocessing_settings = json.load(open(signal['preprocessing_settings']))
    stft_args = preprocessing_settings['processing_parameters']['stft_args']
    preemphasis_coeff = preprocessing_settings['processing_parameters']['preemphasis_coeff']
    n_fft = 2048
    if 'n_fft' in stft_args:
        n_fft = stft_args['n_fft']

    params['rng'] = np.random.RandomState(random_seed)

    if read_checkpoint:
        model.load(library_checkpoint_file)

    signals = []
    noises = []
    signal_costs = []
    noise_costs = []
    for j in range(num_batches):
        for i in range(batch_size):
            spec, bin_masks, source_specs, uids, snrs = next(mixer)

            model_spec = spec
            spec = spec[0]
            bin_masks = bin_masks[0]
            source_specs = source_specs[0]
            uids = uids[0]
            snrs = snrs[0]
            
            # isolate the relavant parts of the sources
            m = model.temporal_mask_for_best_features(source_specs[0])
            signal_splits = model.split_along_time_bins(source_specs[0], m)
            for signal_split in signal_splits:
                signals.append(signal_split)


            m = model.temporal_mask_for_best_features(source_specs[1])
            noise_splits = model.split_along_time_bins(source_specs[1], m)
            for noise_split in noise_splits:
                noises.append(noise_split)

        #H, W, obj = model.update(source_specs[0], 'signal',
        #                         max_iter=params['max_iter'], conv_eps=params['conv_eps'],
        #                         rng=params['rng'],
        #                         verbose=params['verbose'])
        #print('iteration {}: final signal cost {}'.format(i + 1, obj['cost'][-1]))

        #H, W, obj = model.update(source_specs[1], 'noise',
        #                         max_iter=params['max_iter'], conv_eps=params['conv_eps'],
        #                         rng=params['rng'],
        #                         verbose=params['verbose'])
        #print('iteration {}: final noise cost {}'.format(i + 1, obj['cost'][-1]))


        H, W, obj = model.batch_update(signals, 'signal',
                                       max_iter=params['max_iter'], conv_eps=params['conv_eps'],
                                       rng=params['rng'],
                                       verbose=params['verbose'])
    
        H, W, obj = model.batch_update(noises, 'noise',
                                       max_iter=params['max_iter'], conv_eps=params['conv_eps'],
                                       rng=params['rng'],
                                       verbose=params['verbose'])
    
    model.save(library_output_file)


if __name__ == '__main__':
    main()
