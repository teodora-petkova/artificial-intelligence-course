import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

"""
Discrete Fourier Transform 1D
"""
def dft1D(x, sampling_points):
    N = len(x)
    X = np.array([])
    for k in range(0, sampling_points): 
        Xk = np.complex(0, 0)
        for n in range(0, N): # [0, N-1]
            alpha = 2*math.pi*k*n/N
            Xk +=  np.complex(x[n]) * np.complex(math.cos(alpha), (- math.sin(alpha)))
        X = np.append(X, [Xk])
    return X

"""
Inverse Discrete Fourier Transform 1D
"""
def idft1D(X, N): 
    sampling_points = len(X)
    x = np.array([])
    for n in range(0, N): # [0, N-1]
        xn = 0
        for k in range(0, sampling_points): 
            alpha = 2*math.pi*k*n/N
            xn += X[k] * np.complex(math.cos(alpha), math.sin(alpha))
        x = np.append(x, [(1/N) * xn])
    return x

"""
Extract sines from a frequency signal
"""
def extract_sines(possible_frequencies, dft_frequencies):
    functions = np.array([])
    for (frequency, dft) in zip(possible_frequencies, dft_frequencies):
        amplitude = math.sqrt(dft.real**2 + dft.imag**2)
        if(amplitude > 0.000001):
            phase = 0 #math.atan2(-dft.imag, dft.real) * np.pi #if dft.imag != 0 else 0
            print(phase)
            # break the direct tie of the variables inside the main lambda
            # -> wrap them in another lambda called in the loop
            sine = lambda a, p, f: (lambda t : ((1/a) * math.sin(p + f * 2 * math.pi * t)))
            wrapped_sine = np.vectorize((sine)(amplitude, phase, frequency))
            functions = np.append(functions, [wrapped_sine])
    return functions

def test_dft1D():
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

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    #ax1.xlabel("Amplitude")
    #ax1.ylabel("Time [s]")
    ax1.plot(t, y)
    #ax2.xticks([np.arange(0, 1, 1)])
    ax2.bar(f[:N//2], np.abs(dft)[:N//2] * 2/N, width=0.3)

    for sine in extract_sines(f[:N//2], dft[:N//2] * 2/N):
        ax3.plot(t, sine(t))
    ax4.plot(t, idft)
    plt.show()
 
def check_correctness_1D():
    N = 100
    x = np.random.random(N)
    dft = dft1D(x, N)
    fft = np.fft.fft(x)
    print(np.allclose(dft, fft))
    print(np.allclose(idft1D(dft, N), np.fft.ifft(fft)))

"""
Discrete Fourier Transform 2D

1.) do 1D DFT on each row (real to complex)
the first step yields an intermediary 'picture' in which
the horizontal axis is frequency f and the vertical axis is space y
2.) do 1D DFT on each column of the result (complex to complex)
the second step is to apply 1D Fourier transform individually to 
the vertical line of the intermediate image
"""
def dft2D(image):
    M, N = image.shape
    frequency_image = np.empty((M, N), dtype=complex)
    
    # DFT 1D on rows
    for m in range(0, M):
        frequency_image[m, :] = dft1D(image[m, :], N)
    
    # DFT 1D on columns
    for n in range(0, N):
        frequency_image[:, n] = dft1D(frequency_image[:, n], M)
        
    return frequency_image

"""
Inverse Discrete Fourier Transform 2D
"""
def idft2D(frequency_image):
    M, N = frequency_image.shape
    image = np.empty((M, N), dtype=complex)
    
    # IDFT 1D on rows
    for m in range(0, M):
        image[m, :] = idft1D(frequency_image[m, :], N)
    
    # IDFT 1D on columns
    for n in range(0, N):
        image[:, n] = idft1D(image[:, n], M)
        
    return image

def display(image):
        # if there is only one channel to show, display it as grayscale",
        cm = None
        if(len(image.shape)) == 2:
            cm = "gray"
        plt.figure(figsize = (5, 10))
        plt.imshow(image, cmap = cm)
        plt.xticks([])
        plt.yticks([])
        plt.show()

def test_dft2D_with_small_image_pulse():
    original_image = np.array(
        [[0,0,0,0],
         [0,1,1,0],
         [0,1,1,0],
         [0,0,0,0]])
    
    display(original_image)
    dft = dft2D(original_image)
    display(np.abs(dft))
    idft = idft2D(dft)
    display(np.abs(idft))

def check_correctness_2D():
    m = np.random.rand(30, 30)
    dft = dft2D(m)
    fft = np.fft.fft2(m)
    print(np.allclose(dft, fft))
    print(np.allclose(idft2D(dft), np.fft.ifft2(dft)))

test_dft1D()
check_correctness_1D()
test_dft2D_with_small_image_pulse()
check_correctness_2D()