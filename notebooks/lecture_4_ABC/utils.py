import matplotlib.pyplot as plt
import numpy as np

STYLE_THROW = dict(xlim=(0,150), ylim=(0,40), xlabel='Distance traveled (m)', ylabel='Height (m)')
start_bold, end_bold = '\033[1m', '\033[0m'

STYLE_LOTKA = dict(xlim=(0,20), ylim=(0,50), xlabel='Time', ylabel='Population')
start_bold, end_bold = '\033[1m', '\033[0m'

def plot_throw2(d_o, x_o, d, x, color_1='black', color_2='red', 
             style=STYLE_THROW, grid=True, legend=True, ax=None):
    """Return axis comparing simulation output to observations. Show obs. error.
    
    Creates axis if not passed."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(d_o, x_o, label='$x_\mathrm{o}$', marker='.', color=color_1, s=10)
    ax.scatter(d, x, label='s', marker='o', color=color_2, s=15,facecolors='none')
    
    style['title'] = f"MSE: {np.square(np.subtract(x_o[x_o>0], x[x_o>0])).mean():.2f}"
        
    plt.setp(ax, **style)
    if legend:
        ax.legend()
    ax.grid(True)
    
    return ax

def plot_lotka(t, rabbits, foxes, color_1='blue', color_2='red', ax=None, style=STYLE_LOTKA):
    
    """Return axis comparing simulation output to observations. Show parameters.
    
    Creates axis if not passed."""
    
    # Creates axis if not passed."""
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(t, rabbits, 'r-', label='Rabbits')
    ax.plot(t, foxes  , 'b-', label='Foxes')
    ax.grid(True)
    plt.legend(loc='best')
    plt.setp(ax, **style)
    
    return ax