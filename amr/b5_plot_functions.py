import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch


def box_mesh(var, level, box):
    """Get the x y meshgrid for one box."""

    X, Y = np.meshgrid(
        np.concatenate(
            (
                var.x[level][box] - var.dx[level] / 2.0,
                [var.x[level][box][-1] + var.dx[level] / 2.0],
            )
        ),
        np.concatenate(
            (
                var.y[level][box] - var.dx[level] / 2.0,
                [var.y[level][box][-1] + var.dx[level] / 2.0],
            )
        ),
    )
    return X, Y


def box_plot(var, level, box, ax, cmap, vmin, vmax):
    """Plot the data for one box"""

    X, Y = box_mesh(var, level, box)
    pcm = ax.pcolormesh(X, Y, var.data[level][box], cmap=cmap, vmin=vmin, vmax=vmax)
    return pcm


def box_rect(var, level, box, ax):
    """Plot the rectangle for one box"""

    color = ["k", "r", "b", "g", "m", "c"]
    rect = ptch.Rectangle(
        (var.x[level][box][0:2].sum() / 2.0, var.y[level][box][0:2].sum() / 2.0),
        var.x[level][box][-2] - var.x[level][box][0],
        var.y[level][box][-2] - var.y[level][box][0],
        linewidth=1,
        edgecolor=color[level],
        facecolor="none",
    )
    ax.add_patch(rect)
    return rect


def box_umod(u, v, level, box):
    """Calculate the velocity magnitude for one box"""

    umod = (
        u.data[level][box] * u.data[level][box]
        + v.data[level][box] * v.data[level][box]
    ) ** 0.5
    return umod


def plot_b5(var, cmap, vmin, vmax):
    """Create plot for variable in a bisicles plot file.
    var: variable to be plotted (i.e thickness)
    cmap: colormap to be used (same as when plotting with matplotlib)
    vmin, vmax: minimum and maximum extent (same as matplotlib)"""

    fig, ax = plt.subplots()
    for level in range(var.levels):
        for box in range(var.boxes[level]):
            pcm = box_plot(var, level, box, ax, cmap, vmin, vmax)
            # rect = box_rect(var, level, box, ax)
    return fig, ax, pcm


def plot_umod(u, v, cmap, vmin, vmax):
    """Create velocity magnitude plot for a bisicles plot file.
    u, v: x and y velocity components
    cmap: colormap to be used (same as when plotting with matplotlib)
    vmin, vmax: minimum and maximum extent (same as matplotlib)"""

    fig, ax = plt.subplots()
    for level in range(u.levels):
        for box in range(u.boxes[level]):
            umod = np.log10(box_umod(u, v, level, box))
            X, Y = box_mesh(u, level, box)
            pcm = ax.pcolormesh(X, Y, umod, cmap=cmap, vmin=vmin, vmax=vmax)
    return fig, ax, pcm
