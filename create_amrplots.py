import os
from glob import iglob
import matplotlib.pyplot as plt
from amr import bisiclesh5 as b5
from amr import b5_plot_functions as b5plot


path = r'/nobackup/users/donnelly/Antarctica/LRF8'


def open_file(file):
    dhdt = b5.bisicles_var(file, 5)
    xvel = b5.bisicles_var(file, 1)
    yvel = b5.bisicles_var(file, 2)
    return dhdt, xvel, yvel

def plot_vars(dhdt, xvel, yvel, title):
    fig, ax, pcm = b5plot.plot_umod(xvel, yvel, "jet", 0, 4)
    cb = fig.colorbar(
        pcm, ticks=[0, 1, 2, 3, 4], orientation="vertical", pad=0.05, shrink=0.7
    )
    cb.set_label(r"Velocity ($\mathregular{ma^{-1}}$)")
    cb.ax.tick_params(labelsize=12)
    plt.title(title)
    ax.set_aspect("equal", "box")
    fig.set_size_inches(8, 8)
    fig.tight_layout()
    plt.savefig(title + "_umod.png")

    fig, ax, pcm = b5plot.plot_b5(dhdt, "seismic", -5, 5)
    cb = fig.colorbar(pcm, orientation="vertical", pad=0.05, shrink=0.7)
    cb.set_label(r"Rate of Thickness Change ($\mathregular{ma^{-1}}$)")
    cb.ax.tick_params(labelsize=12)
    plt.title(title)
    ax.set_aspect("equal", "box")
    fig.set_size_inches(8, 8)
    fig.tight_layout()
    plt.savefig(title + "_dhdt.png")

if __name__ == "__main__":
    files =sorted(iglob(os.path.join(path, "plot.*")), reverse=True)
    last_plot = files[0]
    half_plot = files[100]
    first_plot = files[199]

    dhdt, xvel, yvel = open_file(first_plot)
    plot_vars(dhdt, xvel, yvel, "LRF0")

    dhdt100, xvel100, yvel100 = open_file(half_plot)
    plot_vars(dhdt100, xvel100, yvel100, "LRF100")

    dhdt200, xvel200, yvel200 = open_file(last_plot)
    plot_vars(dhdt200, xvel200, yvel200, "LRF200")