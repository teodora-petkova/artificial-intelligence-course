import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from collections import namedtuple

"""
Discrete Fourier Transform 1D
"""


def dft1D(x):
    N = len(x)
    X = np.array([])
    for k in range(0, N):
        Xk = np.complex(0, 0)
        for n in range(0, N):  # [0, N-1]
            alpha = 2*math.pi*k*n/N
            Xk += np.complex(x[n]) * \
                np.complex(math.cos(alpha), (- math.sin(alpha)))
        X = np.append(X, [Xk])
    return X


"""
Inverse Discrete Fourier Transform 1D
"""


def idft1D(X):
    N = len(X)
    x = np.array([])
    for n in range(0, N):  # [0, N-1]
        xn = 0
        for k in range(0, N):
            alpha = 2*math.pi*k*n/N
            xn += X[k] * np.complex(math.cos(alpha), math.sin(alpha))
        x = np.append(x, [(1/N) * xn])
    return x


"""
Fast Fourier Transform
"""


def fft1D(x):
    N = len(x)
    # if N % 2 > 0:
    #    raise ValueError("The input must be a power of 2")
    if(N <= 1):
        return x
    M = (int)(N/2)
    even = np.array(fft1D(x[0::2]))
    odd = np.array(fft1D(x[1::2]))

    frequency_bins = np.zeros(N, dtype=complex)
    for k in range(0, M):
        alpha = 2*math.pi*k/N
        twiddle_factor = np.complex(math.cos(alpha), (- math.sin(alpha)))
        oddTerm = twiddle_factor*odd[k]
        frequency_bins[k] = even[k] + oddTerm
        frequency_bins[k+M] = even[k] - oddTerm
    return frequency_bins


"""
Inverse Fast Fourier Transform
"""


def ifft1D(freqs):
    N = len(freqs)
    if(N <= 1):
        return freqs
    M = (int)(N/2)
    even = np.array(ifft1D(freqs[0::2]))
    odd = np.array(ifft1D(freqs[1::2]))

    x = np.zeros(N, dtype=complex)
    for k in range(0, M):
        alpha = 2*math.pi*k/N
        twiddle_factor = np.complex(math.cos(alpha), math.sin(alpha))
        oddTerm = twiddle_factor*odd[k]
        x[k] = (1/2)*(even[k] + oddTerm)
        x[k+M] = (1/2)*(even[k] - oddTerm)
    return x


"""
Extract sines from a frequency signal
"""


def extract_sines(possible_frequencies, dft_frequencies):
    functions = np.array([])
    for (frequency, dft) in zip(possible_frequencies, dft_frequencies):
        amplitude = math.sqrt(dft.real**2 + dft.imag**2)
        if(amplitude > 0.000001):
            # math.atan2(-dft.imag, dft.real) * np.pi #if dft.imag != 0 else 0
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
    #fft = np.fft.fft(y, N)
    idft = idft1D(dft)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    # ax1.xlabel("Amplitude")
    #ax1.ylabel("Time [s]")
    ax1.plot(t, y)
    #ax2.xticks([np.arange(0, 1, 1)])
    ax2.bar(f[:N//2], np.abs(dft)[:N//2] * 2/N, width=0.3)

    for sine in extract_sines(f[:N//2], dft[:N//2] * 2/N):
        ax3.plot(t, sine(t))
    ax4.plot(t, idft)


"""
Discrete Fourier Transform 2D

1.) do 1D DFT on each row (real to complex)
the first step yields an intermediary 'picture' in which
the horizontal axis is frequency f and the vertical axis is space y
2.) do 1D DFT on each column of the result (complex to complex)
the second step is to apply 1D Fourier transform individually to 
the vertical line of the intermediate image
"""


def ft2D(image, ft):
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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
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


"""
def fftshift(m):
    sz = ceil(size(A)/2)
    A = A([sz(1)+1:end, 1:sz(1)], [sz(2)+1:end, 1:sz(2)])
"""

Point = namedtuple('Point', 'x y')


def distance(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)


def filter_image(image, filter, update):
    rows, columns = image.shape
    center = Point(rows/2, columns/2)
    for r in range(0, rows):
        for c in range(0, columns):
            if(filter(distance(Point(r, c), center))):
                update(image, c, r)
    return image


def high_pass_fiter(image, threshold=50):
    def hp(dist):
        return dist < threshold

    def update(image, c, r):
        image[c][r] = 0
    return filter_image(image, hp, update)


def low_pass_fiter(image, threshold=50):
    def lp(dist):
        return dist >= threshold

    def update(image, c, r):
        image[c][r] = 1
    return filter_image(image, lp, update)


def test_fft2D_with_photo():
    original_image = imread("https://www.hlevkin.com/TestImages/cameraman.bmp")
    fft = fft2D(original_image)
    shifted_fft = np.fft.fftshift(fft)
    hp_fft = high_pass_fiter(shifted_fft.copy())
    lp_fft = low_pass_fiter(shifted_fft.copy())
    hp = ifft2D(hp_fft)
    lp = ifft2D(lp_fft)

    cm = "gray"
    _, ((ax_orig, ax_hp, ax_lp), (ax_orig_freqs, ax_hp_freqs,
                                  ax_lp_freqs)) = plt.subplots(2, 3, figsize=(10, 10))
    ax_orig.imshow(original_image, cmap=cm)
    ax_orig.set_title("Original Image")
    ax_hp.imshow(np.abs(hp), cmap=cm)
    ax_hp.set_title("High Pass Filter")
    ax_lp.imshow(np.abs(lp), cmap=cm)
    ax_lp.set_title("Low Pass Filter")

    ax_orig_freqs.imshow(shiftlog(fft), cmap=cm)
    ax_orig_freqs.set_title("Frequency spectrum FFT")
    ax_hp_freqs.imshow(logabs(hp_fft), cmap=cm)
    ax_hp_freqs.set_title("Frequency spectrum High Pass Filter")
    ax_lp_freqs.imshow(logabs(lp_fft), cmap=cm)
    ax_lp_freqs.set_title("Frequency spectrum Low Pass Filter")

    plt.setp([ax_orig, ax_hp, ax_lp], xticks=[], yticks=[])
    # TODO: correct ticks for the other axis -> (0, 0) in the middle


def check_correctness_1D():
    N = 32
    x = np.random.random(N)
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
test_dft1D()
test_dft2D_with_small_image_pulse()
test_fft2D_with_photo()
plt.show()
