import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from amr import bisiclesh5 as b5
from amr import b5_plot_functions as b5plot


file = '/net/pc200261/nobackup/users/donnelly/Antarctica/picontrol/lev-grid/plot.pi_ant.011705.2d.hdf5'
thk =  b5.bisicles_var(file, 0)
dhdt =  b5.bisicles_var(file, 5)
xvel =  b5.bisicles_var(file, 1)
yvel =  b5.bisicles_var(file, 2)

fig, ax, pcm = b5plot.plot_umod(xvel,yvel,'jet',0,4)
cb = fig.colorbar(pcm, ticks=[0,1,2,3,4], orientation = 'vertical', pad=0.05, shrink = 0.7)
cb.set_label(r'Velocity ($\mathregular{ma^{-1}}$)')
cb.ax.tick_params(labelsize=12) 
plt.title("SSP5 8.5 t=2025")
ax.set_aspect('equal','box')
fig.set_size_inches(8, 8)
fig.tight_layout()
plt.savefig("example_umod.png")

fig, ax, pcm = b5plot.plot_b5(dhdt, 'seismic', -5, 5)
cb = fig.colorbar(pcm, orientation = 'vertical', pad=0.05, shrink = 0.7)
cb.set_label(r'Rate of Thickness Change ($\mathregular{ma^{-1}}$)')
cb.ax.tick_params(labelsize=12) 
plt.title("Example plot")
ax.set_aspect('equal','box')
fig.set_size_inches(8, 8)
fig.tight_layout()
plt.savefig("example_dhdt.png")