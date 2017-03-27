## Matrix Factorization 

Matrix factorization using various simple components analysis algorithms. The iPython notebooks are on random signals and some audio signals. Here are libraries using these, foremost are PCA, ICA, and we're working on NMF.

Current libraries:
- _PCA_ - (Principle Component Analysis) in [pca.py](pca.py) 
- _ICA_ - (Independent Component Analsys) in [ica.py](ica.py) 
- _NMF_ - (Non-negative Matrix Factorization) in [nmf.py](nmf.py) 
  - L-1 normalization (to add...)
  - L-2 normalization (to add...)

### Independent Component Analysis

All notebooks are standalone at present. Their descriptions are as follows:

| iPython Notebook            | Audio Sources              | Channels / Mixing | Software Packages | Time/Freq Domain |
| ----------------            | ----------------           | ---------------- | ---------------- | ---------------- |
|`main-ica.ipynb`             | Synthesized (Saw/Sin/Gaus) | Synthesized (Gaussian) | scikit-learn's toolbox | Time-domain |
|`main-audio-ica.ipynb`       | 20-Source Single Audio     | Synthesized (Gaussian) | scikit-learn's toolbox | |
|`main-real-ica.ipynb`        | 5-Source Mixed Audio       | 5 iPhones | scikit-learn's toolbox | |
|`implemented-fastica.ipynb`  | Synthesized waveforms      | 5 iPhones | Implemented from [Wikipedia](https://en.wikipedia.org/wiki/FastICA) | |
|`implemented-nmf.ipynb`      | 2-Source Mixed Audio       | 1 Channel | Implemented from [Stanford slides](https://ccrma.stanford.edu/~njb/teaching/sstutorial/part2.pdf) | STFT |

