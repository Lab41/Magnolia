import numpy as np
from scipy.signal import sawtooth

def gennoisy(S=None, A=None, N=None):
    
    # A is the channel
    if not A:
        A = np.random.rand(3,3);
        # A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]]) 
        
    # Generate the signal
    if not S:
        # Time with sampling frequency 200 Hz
        fs = 1.0/200;
        t = np.zeros(2000);
        t[:] = np.arange(0,10,fs);
        
        # Signal
        # s1 = np.sin( 2*np.pi*t );
        # s2 = 0*sawtooth(2*np.pi*10*t);
        # s3 = np.sin( 2*np.pi*t + np.pi/2.);
        s1 = np.sin(2 * t)  # Signal 1 : sinusoidal signal
        s2 = np.sign(np.sin(3 * t))  # Signal 2 : square signal
        s3 = sawtooth(2 * np.pi * t)  # Signal 3: saw tooth signal

        S = np.array([s1,s2,s3]).T;

    # Generate the noise
    if not N:
        N = .05*np.random.randn(*S.shape);

    # Final microphone has elements of each signal
    X = S.dot(A)+N;

    return X, S, A, N
