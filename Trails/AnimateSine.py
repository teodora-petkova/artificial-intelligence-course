import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import IPython.display as display

def animate_sine():
    min_x = -3*math.pi
    max_x = 3*math.pi

    def g(n):
        return (np.exp(2*math.pi*1j*n/10)) * np.sin(n)
    x = np.linspace(min_x, max_x, 350)
    
    figure, _ = plt.subplots()
    p = g(x)
    line, = plt.plot([], [], 'g-', animated=True)
    point, = plt.plot(p.real, p.imag, "ro", animated=True)

    fig = plt.gcf()
    fig.gca().set_aspect('equal')
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.plot(np.real(g(x)), np.imag(g(x)))

    def update(x):
        p = g(x)
        line_x = np.linspace(0, p.real)
        line_y = (p.imag/p.real)*line_x if p.real != 0 else 0
        line.set_data(line_x, line_y)
        point.set_data(p.real, p.imag)
        return point, line

    anim = animation.FuncAnimation(figure,
                                   update,
                                   frames=np.linspace(min_x, max_x, 400, endpoint=False),
                                   interval=10,
                                   blit=True)
    return anim

anim = animate_sine()
#display.HTML(anim.to_jshtml())
#display.HTML(anim.to_html5_video())
plt.show()
