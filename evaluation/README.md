# Evaluation

Evaluation source and example notebooks should be available here. Given a reference signal and an estimated signal, we evaluate on several metrics. Included in the literature are:

- Peak Signal to Noise Ratio (PSNR)
- PESQ's Mean Opinion Score (MOS), perhaps not able to be used
- Squared error (equivalent to SNR)

## Procedure to evaluate your signals

We will settle on training, test, and dev partitions from specific datasets. Each specific set of files will be representative of performance across all signals. Please train on the training set, validate on the dev dataset, and report metrics on the test dataset.

Evaluation will require reference (truth) signal and predicted signal. These may require alignment.

