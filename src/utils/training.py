import numpy as np


def preprocess_l41_batch(spec_batch, mask_batch=None):
    # should be dimensions of (batch size, time frame, frequency)
    spec_batch = spec_batch.transpose(0, 2, 1)
    spec_batch = scale_input_spectrogram_for_l41_model(spec_batch)

    if mask_batch is not None:
        # should be dimensions of (batch size, time frame, frequency, source)
        mask_batch = mask_batch.transpose(0, 3, 2, 1)
        mask_batch = convert_boolean_mask_for_l41_model(mask_batch)
        return spec_batch, mask_batch

    return spec_batch


def scale_input_spectrogram_for_l41_model(spec_batch):
    spec_batch = np.sqrt(np.abs(spec_batch))
    return (spec_batch - spec_batch.min())/(spec_batch.max() - spec_batch.min())


def convert_boolean_mask_for_l41_model(mask_batch):
    return 2.0*mask_batch.astype(float) - 1.0
