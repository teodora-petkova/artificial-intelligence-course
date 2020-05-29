import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from collections import namedtuple

def dft1D_base(x, sign, factor):
    """
    Discrete Fourier Transform 1D
    """
    N = len(x)
    X = np.array([])
    for k in range(0, N):
        Xk = np.complex(0, 0)
        for n in range(0, N):
            exp = np.exp(sign*2*math.pi*k*n*1j/N)
            Xk += np.complex(x[n]) * exp
        X = np.append(X, [factor * Xk])
    return X

def dft1D(x):
    return dft1D_base(x, -1, 1)

def idft1D(X):
    return dft1D_base(X, 1, 1/len(X))

def fft1D_base(x, sign, factor, ft1D):
    # the input must be a power of 2
    # in order to work with arbitrary dimensions
    # to implement padding
    N = len(x)
    if(N <= 1):
        return x
    M = (int)(N/2)
    even = np.array(ft1D(x[0::2]))
    odd = np.array(ft1D(x[1::2]))

    X = np.zeros(N, dtype=complex)
    for k in range(0, M):
        exp = np.exp(sign*2*math.pi*k*1j/N)
        oddTerm = exp*odd[k]
        X[k] = factor*(even[k] + oddTerm)
        X[k+M] = factor*(even[k] - oddTerm)
    return X

def fft1D(x):
    return fft1D_base(x, -1, 1, fft1D)

def ifft1D(freqs):
    return fft1D_base(freqs, 1, 0.5, ifft1D)

def extract_sines(possible_frequencies, dft_frequencies):
    """
    Extract sines from a frequency signal
    """
    functions = np.array([])
    for (frequency, dft) in zip(possible_frequencies, dft_frequencies):
        amplitude = math.sqrt(dft.real**2 + dft.imag**2)
        if(amplitude > 0.000001):
            #phase = math.atan2(-dft.imag, dft.real) * np.pi if dft.imag != 0 else 0
            #print(phase)
            phase = 0
            # print(phase)
            # break the direct tie of the variables inside the main lambda
            # -> wrap them in another lambda called in the loop
            def sine(a, p, f): return (lambda t: (
                (1/a) * math.sin(p + f*2*math.pi*t)))
            wrapped_sine = np.vectorize((sine)(amplitude, phase, frequency))
            functions = np.append(functions, [wrapped_sine])
    return functions

def test_dft1D():
    time_period = 1  # 1 second
    Fs = 100.0  # sampling rate - sampling rate is the number of samples per second
    # it is the reciprocal of the sampling time, i.e. 1/T, also called the sampling frequency

    Ts = time_period/Fs  # sampling interval / sampling period
    # the sampling time is the time interval between successive samples, also called the sampling interval or the sampling period

    t = np.arange(0, time_period, Ts)  # time vector in seconds
    N = t.size
    f = np.linspace(0, 1/Ts, N)  # frequency vector - 1 / sampling_interval

    cycle = 2*np.pi
    frequency1 = 1 * cycle  # frequency of the signal 1 in Hertz
    frequency2 = 3 * cycle  # frequency of the signal 3 in Hertz

    y = np.sin(t*frequency1) + np.sin(t*frequency2)
    dft = dft1D(y)
    idft = idft1D(dft)

    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 100))
    
    ax1.plot(t, y)
    ax1.set_title("Signal in the time domain")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")

    ax2.bar(f[:N//2], np.abs(dft)[:N//2] * 2/N, width=0.3)
    ax2.set_title("Signal in the frequency domain")
    ax1.set_xlabel("Frequency [Hertz]")
    ax1.set_ylabel("Amplitude")

    for sine in extract_sines(f[:N//2], dft[:N//2] * 2/N):
        ax3.plot(t, sine(t))
    ax3.set_title("Extracted Signals in the time domain")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Amplitude")
    
    ax4.plot(t, idft)
    ax4.set_title("Signal retrieved by the inverse DFT")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Amplitude")

def ft2D(image, ft):
    """
    Discrete Fourier Transform 2D

    1.) do 1D DFT on each row (real to complex)
    the first step yields an intermediary 'picture' in which
    the horizontal axis is frequency f and the vertical axis is space y
    2.) do 1D DFT on each column of the result (complex to complex)
    the second step is to apply 1D Fourier transform individually to
    the vertical line of the intermediate image 
    """
    M, N = image.shape
    transformed_image = np.empty((M, N), dtype=complex)

    # DFT 1D on rows
    for m in range(0, M):
        transformed_image[m, :] = ft(image[m, :])

    # DFT 1D on columns
    for n in range(0, N):
        transformed_image[:, n] = ft(transformed_image[:, n])

    return transformed_image

def dft2D(image):
    return ft2D(image, dft1D)

def idft2D(image):
    return ft2D(image, idft1D)

def fft2D(image):
    return ft2D(image, fft1D)

def ifft2D(image):
    return ft2D(image, ifft1D)

def test_dft2D_with_small_image_pulse():
    original_image = np.array(
        [[0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]])

    dft = dft2D(original_image)
    idft = idft2D(dft)

    cm = "gray"
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    ax1.imshow(original_image, cmap=cm)
    ax1.set_title("Original Image")
    ax2.imshow(np.abs(dft), cmap=cm)
    ax2.set_title("Frequency spectrum DFT")
    ax3.imshow(np.abs(idft), cmap=cm)
    ax3.set_title("Image with inverse DFT")

def shiftlog(m):
    return np.fft.fftshift(np.log(1+np.abs(m)))

def logabs(m):
    return np.log(1+np.abs(m))

Point = namedtuple('Point', 'x y')

def distance(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

def image_filter(image, condition, default_value, filter_value):
    (columns, rows) = image.shape
    center = Point(rows/2, columns/2)

    base = []
    if default_value == 1:
        base = np.ones((columns, rows))
    else:
        base = np.zeros((columns, rows))

    for r in range(0, rows):
        for c in range(0, columns):
            if(condition(distance(Point(r, c), center))):
                base[c, r] = filter_value
    return base

def high_pass_filter(image, threshold=50):
    def hp(dist):
        return dist < threshold
    return image_filter(image, hp, 1, 0)

def low_pass_filter(image, threshold=50):
    def lp(dist):
        return dist < threshold
    return image_filter(image, lp, 0, 1)

def custom_noise_filter(image,
                 notch_size,
                 threshold_for_unusual_peaks,
                 threshold_for_lp=50):

    columns, rows = image.shape
    base = np.ones((columns, rows))
    center = Point(rows/2, columns/2)
    notch_half = notch_size//2
    for r in range(0, rows):
        for c in range(0, columns):
            if(distance(Point(r, c), center) >= threshold_for_lp and
                    np.abs(image[c, r]) > threshold_for_unusual_peaks):
                for n in range(max(r-notch_half, 0), min(r+notch_half, rows)):
                    for m in range(max(c-notch_half,0), min(c+notch_half, columns)):
                        base[m, n] = 0
    return base

def test_fft2D_with_sinusoids(frequency1, frequency2, title=""):
    columns, rows = 128, 128
    x = np.linspace(0, 1, rows)
    y = np.linspace(0, 1, rows)
    X = np.repeat(x[np.newaxis, :], rows, axis=0)
    Y = np.repeat(y[:, np.newaxis], columns, axis=1)
    sinusoid = np.sin(frequency1*2*np.pi*X) + np.sin(frequency2*2*np.pi*Y)

    fft = fft2D(sinusoid)

    cm = "gray"
    _, (ax_orig, ax_freq) = plt.subplots(1, 2, figsize=(10, 10))
    ax_orig.imshow(sinusoid, cmap=cm)
    ax_orig.set_title("Sinusoid: %s" % title)

    square = 150
    # zoom-in
    ax_freq.imshow(np.fft.fftshift(np.abs(fft))[-square:square, -square:square],
                   cmap=cm, extent=[-square/2, square/2, -square/2, square/2])
    ax_freq.set_title("Zoomed-in frequency spectrum FFT")

    plt.setp([ax_orig], xticks=[], yticks=[])

def test_fft2D_with_hp_and_lp_filters():
    original_image = imread("https://www.hlevkin.com/TestImages/cameraman.bmp")
    fft = fft2D(original_image)
    shifted_fft = np.fft.fftshift(fft)
    hp_fft = shifted_fft * high_pass_filter(shifted_fft.copy())
    lp_fft = shifted_fft * low_pass_filter(shifted_fft.copy())
    hp = ifft2D(hp_fft)
    lp = ifft2D(lp_fft)

    (rows, columns) = original_image.shape
    rows_half = (int)(rows/2)
    columns_half = (int)(columns/2)
    cm = "gray"
    _, ((ax_orig, ax_hp, ax_lp), 
        (ax_orig_freqs, ax_hp_freqs, ax_lp_freqs)) = plt.subplots(2, 3, figsize=(10, 10))
    ax_orig.imshow(original_image, cmap=cm)
    ax_orig.set_title("Original Image")
    ax_hp.imshow(np.abs(hp), cmap=cm)
    ax_hp.set_title("High Pass Filter")
    ax_lp.imshow(np.abs(lp), cmap=cm)
    ax_lp.set_title("Low Pass Filter")

    ax_orig_freqs.imshow(shiftlog(fft), cmap=cm,
        extent=[-columns_half, columns_half, -rows_half, rows_half])
    ax_orig_freqs.set_title("Frequency spectrum FFT")
    ax_hp_freqs.imshow(logabs(hp_fft), cmap=cm, \
        extent=[-columns_half, columns_half, -rows_half, rows_half])
    ax_hp_freqs.set_title("Frequency High Pass Filter")
    ax_lp_freqs.imshow(logabs(lp_fft), cmap=cm, \
        extent=[-columns_half, columns_half, -rows_half, rows_half])
    ax_lp_freqs.set_title("Frequency Low Pass Filter")

    plt.setp([ax_orig, ax_hp, ax_lp], xticks=[], yticks=[])

def test_fft2D_with_noise_removal():
    original_image = imread(r"moonlanding.png")
    fft = fft2D(original_image)
    shifted_fft = np.fft.fftshift(fft)
    noise_filter = custom_noise_filter(shifted_fft.copy(),
                                notch_size=9,
                                threshold_for_unusual_peaks=300000)

    filtered_fft = shifted_fft * noise_filter

    ifft = ifft2D(filtered_fft)

    (rows, columns) = filtered_fft.shape
    rows_half = (int)(rows/2)
    columns_half = (int)(columns/2)

    cm = "gray"
    _, ((ax_orig, ax_filtered), (ax_orig_freqs, ax_filtered_freqs)
        ) = plt.subplots(2, 2, figsize=(10, 10))
    ax_orig.imshow(original_image, cmap=cm)
    ax_orig.set_title("Original Image")
    ax_filtered.imshow(np.abs(ifft), cmap=cm)
    ax_filtered.set_title("Removed noise")

    ax_orig_freqs.imshow(shiftlog(fft), cmap=cm, 
        extent=[-rows_half, rows_half, -columns_half, columns_half])
    ax_orig_freqs.set_title("Frequency spectrum FFT")
    ax_filtered_freqs.imshow(logabs(filtered_fft), cmap=cm,
        extent=[-rows_half, rows_half, -columns_half, columns_half])
    ax_filtered_freqs.set_title("Noise Removal Filter")

    plt.setp([ax_orig, ax_filtered], xticks=[], yticks=[])

def check_correctness_1D():
    N = 32
    x = np.random.rand(N)
    dft = dft1D(x)
    fft = np.fft.fft(x)
    custom_fft = fft1D(x)
    ifft = np.fft.ifft(fft)
    print("DFT 1D: ", np.allclose(dft, fft))
    print("Inverse DFT 1D: ", np.allclose(idft1D(dft), ifft))
    print("FFT 1D: ", np.allclose(custom_fft, fft))
    print("Inverse FFT 1D: ", np.allclose(ifft1D(custom_fft), ifft))

def check_correctness_2D():
    m = np.random.rand(32, 32)
    dft = dft2D(m)
    custom_fft = fft2D(m)
    fft = np.fft.fft2(m)
    ifft = np.fft.ifft2(dft)
    print("DFT 2D:", np.allclose(dft, fft))
    print("IDFT 2D: ", np.allclose(idft2D(dft), ifft))
    print("FFT 2D: ", np.allclose(custom_fft, fft))
    print("IFFT 2D: ", np.allclose(ifft2D(custom_fft), ifft))

check_correctness_1D()
check_correctness_2D()
#test_dft1D()
#test_dft2D_with_small_image_pulse()
#test_fft2D_with_hp_and_lp_filters()
#test_fft2D_with_sinusoids(4, 0, r"$ sin(4(2\pi)x) $")
#test_fft2D_with_sinusoids(11, 0, r"$ sin(11(2\pi)x) $")
#test_fft2D_with_sinusoids(3, 11, r"$ sin(3(2\pi)x) + sin(11(2\pi)y) $")
test_fft2D_with_noise_removal()
plt.show()
