## Simple Components Analysis

Doing PCA and ICA on random signals and audio signals. All notebooks are standalone at present. Their descriptions are as follows:

### Independent Component Analysis

| iPython Notebook | Audio Sources    | Channels / Mixing | Software Packages |
| ---------------- | ---------------- | ---------------- | ---------------- |
|`main-ica.ipynb`  | Synthesized (Saw/Sin/Gaus) | Synthesized (Gaussian) | scikit-learn's toolbox |
|`main-audio-ica.ipynb`  | 20-Source Single Audio | Synthesized (Gaussian) | scikit-learn's toolbox |
|`main-real-ica.ipynb`  | 5-Source Mixed Audio | 5 iPhones | scikit-learn's toolbox |
|`implemented-fastica.ipynb`  | Synthesized waveforms | 5 iPhones | Implemented from [Wikipedia](https://en.wikipedia.org/wiki/FastICA) |
