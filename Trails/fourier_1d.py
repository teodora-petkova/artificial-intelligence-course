import math
import matplotlib.pyplot as plt
import numpy as np

"""
Inverse Discrete Fourier Transform
"""
def dft1D(x, sampling_points):
    N = len(x)
    X = np.array([])
    for k in range(0, sampling_points): 
        re = 0
        im = 0
        for n in range(0, N): # [0, N-1]
            alpha = 2*math.pi*k*n/N
            re += x[n] * math.cos(alpha)
            im += x[n] * (- math.sin(alpha))
        X = np.append(X, [np.complex(re, im)])
    return X

"""
Inverse Discrete Fourier Transform
"""
def idft1D(X, N): 
    sampling_points = len(X)
    x = np.array([])
    for n in range(0, N): # [0, N-1]
        xn = 0
        for k in range(0, sampling_points): 
            alpha = 2*math.pi*k*n/N
            re = X[k].real * math.cos(alpha) 
            im = X[k].imag * math.sin(alpha)
            xn += (2/N) * (re - im)
        x = np.append(x, [xn])
    return x

"""
Extract sines from a frequency signal
"""
def extract_sines(possible_frequencies, dft_frequencies):
    functions = np.array([])
    for (frequency, dft) in zip(possible_frequencies, dft_frequencies):
        amplitude = math.sqrt(dft.real**2 + dft.imag**2)
        if(amplitude > 0.000001):
            phase = -2 * math.atan(dft.imag/dft.real) if dft.real != 0 else 0
            # break the direct tie of the variables inside the main lambda
            # -> wrap them in another lambda called in the loop
            sine = lambda a, p, f: (lambda t : ((1/a) * math.sin(p + f * 2 * math.pi * t)))
            wrapped_sine = np.vectorize((sine)(amplitude, phase, frequency))
            functions = np.append(functions, [wrapped_sine])
    return functions

def test():
    time_period = 1 # 1 second
    Fs = 100.0;  # sampling rate - sampling rate is the number of samples per second
    # it is the reciprocal of the sampling time, i.e. 1/T, also called the sampling frequency

    Ts = time_period/Fs; # sampling interval / sampling period
    # the sampling time is the time interval between successive samples, also called the sampling interval or the sampling period

    t = np.arange(0, time_period, Ts) # time vector in seconds
    N = t.size
    f = np.linspace(0, 1/Ts, N) # frequency vector - 1 / sampling_interval

    cycle = 2*np.pi
    frequency1 = 1 * cycle # frequency of the signal 1 in Hertz
    frequency2 = 3 * cycle # frequency of the signal 3 in Hertz

    y = np.sin(t*frequency1) + np.sin(t*frequency2)
    dft = dft1D(y, N)
    #fft = np.fft.fft(y, N)
    idft = idft1D(dft, N)

    _, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    #ax1.set_title("A sum of sine waves")
    #ax1.xlabel("Amplitude")
    #ax1.ylabel("Time [s]")
    ax1.plot(t, y)
    #ax2.xticks([np.arange(0, 1, 1)])
    #ax2.set_title("Frequencies of the sine waves by DFT")
    ax2.bar(f[:N//2], np.abs(dft)[:N//2] * 2/N, width=0.3)

    #ax3.set_title("Extracted sine waves by DFT")
    for sine in extract_sines(f[:N//2], dft[:N//2] * 2/N):
        ax3.plot(t, sine(t))

    #ax4.set_title("Inverted DFT - again the initial sum of sine waves")
    ax4.plot(t, idft)
    plt.show()
 
test()