import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import IPython.display as display

def animate_sine():
    min_x = 0
    max_x = 1

    time_period = 1 #seconds
    Fs = 200.0 #sampling rate
    Ts = time_period/Fs # sampling interval

    cycle = 2*np.pi
    frequency = 4 

    def sine_wave(n):
        return np.sin(n*cycle*frequency)

    def g(n):
        return (np.exp(cycle*1j*n)) * sine_wave(n)
    
    x = np.arange(min_x, max_x+Ts, Ts)

    fig, (ax_spiral, ax_sine) = plt.subplots(1, 2, figsize=(10, 40))
    fig.suptitle("The Sine wave with frequency %d Hz wrapped in a spiral" % frequency)

    p = g(x)
    line, = ax_spiral.plot([], [], 'g-', animated=True)
    point_of_spiral, = ax_spiral.plot(p.real, p.imag, "ro", animated=True)

    ax_spiral.set_xlabel("Real")
    ax_spiral.set_ylabel("Imaginary")
    ax_spiral.plot(np.real(g(x)), np.imag(g(x)))

    ax_sine.plot(x, sine_wave(x))
    point_of_sine, = ax_sine.plot(0, 0, "ro", animated=True)

    def update(x):
        p = g(x)
        line_x = np.linspace(0, p.real)
        line_y = (p.imag/p.real)*line_x if p.real != 0 else 0
        line.set_data(line_x, line_y)
        point_of_spiral.set_data(p.real, p.imag)
        point_of_sine.set_data(x, sine_wave(x))
        return point_of_spiral, line, point_of_sine

    anim = animation.FuncAnimation(fig,
                                   update,
                                   frames=np.linspace(min_x, max__x, 80, endpoint=False),
                                   interval=60,
                                   blit=True)
    return anim

anim = animate_sine()
#display.HTML(anim.to_jshtml())
plt.show()
