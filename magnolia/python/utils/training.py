import numpy as np


def preprocess_l41_regression_batch(spec_batch, mask_batch=None, specs_batch=None):
    # should be dimensions of (batch size, time frame, frequency)
    spec_batch = spec_batch.transpose(0, 2, 1)
    scaled_spec_batch = scale_input_spectrogram_for_l41_model(spec_batch)

    if mask_batch is not None and specs_batch is None:
        # should be dimensions of (batch size, time frame, frequency, source)
        mask_batch = mask_batch.transpose(0, 3, 2, 1)
        mask_batch = convert_boolean_mask_for_l41_model(mask_batch)
        return scaled_spec_batch, mask_batch

    if specs_batch is not None and mask_batch is None:
        # should be dimensions of (batch size, time frame, frequency, source)
        specs_batch = specs_batch.transpose(0, 3, 2, 1)
        return scaled_spec_batch, np.abs(specs_batch)

    if specs_batch is not None and mask_batch is not None:
        # should be dimensions of (batch size, time frame, frequency, source)
        mask_batch = mask_batch.transpose(0, 3, 2, 1)
        mask_batch = convert_boolean_mask_for_l41_model(mask_batch)

        # should be dimensions of (batch size, time frame, frequency, source)
        specs_batch = specs_batch.transpose(0, 3, 2, 1)
        return scaled_spec_batch, mask_batch, np.abs(specs_batch)

    return scaled_spec_batch


def preprocess_chimera_batch(spec_batch, mask_batch=None, specs_batch=None):
    # should be dimensions of (batch size, time frame, frequency)
    spec_batch = spec_batch.transpose(0, 2, 1)
    unscaled_spec_batch = np.abs(spec_batch)
    scaled_spec_batch = scale_input_spectrogram_for_l41_model(spec_batch)

    if mask_batch is not None and specs_batch is None:
        # should be dimensions of (batch size, time frame, frequency, source)
        mask_batch = mask_batch.transpose(0, 3, 2, 1)
        # mask_batch = convert_boolean_mask_for_chimera_model(mask_batch)
        return unscaled_spec_batch, scaled_spec_batch, mask_batch

    if specs_batch is not None and mask_batch is None:
        # should be dimensions of (batch size, time frame, frequency, source)
        specs_batch = specs_batch.transpose(0, 3, 2, 1)
        return unscaled_spec_batch, scaled_spec_batch, np.abs(specs_batch)

    if specs_batch is not None and mask_batch is not None:
        # should be dimensions of (batch size, time frame, frequency, source)
        mask_batch = mask_batch.transpose(0, 3, 2, 1)
        # mask_batch = convert_boolean_mask_for_chimera_model(mask_batch)

        # should be dimensions of (batch size, time frame, frequency, source)
        specs_batch = specs_batch.transpose(0, 3, 2, 1)
        return unscaled_spec_batch, scaled_spec_batch, mask_batch, np.abs(specs_batch)

    return unscaled_spec_batch, scaled_spec_batch


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


# def convert_boolean_mask_for_chimera_model(mask_batch):
#     return mask_batch.astype(float)
