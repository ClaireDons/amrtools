# to do
# 1/ rectangle around each box - tick
# 2/ add ghost cells - tick
# 3/ GL locator - tick
# 4/ check relation between grid point and cell - tick
# 5/ create class - tick
# 6/ option to vary level depth of plot as optional input
# 7/ grid structure - which level/box to find each grid point
# 8 add limits and linear/log, colorscale, transform to variable table
# 9 magnitude fn


import h5py
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

from copy import deepcopy

class bisicles_var:

  def __init__(self, fname, tarcomponent):

    def variable_table(vname):

      table = np.array([['thickness','Ice thickness','m'], \
                       ['xVel','Horizontal velocity x-component','m/yr'], \
                       ['yVel','Horizontal velocity y-component','m/yr'], \
                       ['vVel','Vertical velocity','m/yr'], \
                       ['Z_surface','Upper surface elevation','m'], \
                       ['Z_bottom','Lower surface elevation','m'], \
                       ['Z_base','Bedrock elevation','m'], \
                       ['basal_friction','Basal friction','Pa'], \
                       ['C0','-','-'], \
                       ['xRhs','Gravitational driving stress in x','Pa'], \
                       ['yRhs','Gravitational driving stress in y','Pa'], \
                       ['dThickness/dt','Rate of thickness change','m/yr'], \
                       ['xfVel','Horizontal velocity x-component at upper surface','m/yr'], \
                       ['yfVel','Horizontal velocity y-component at upper surface','m/yr'], \
                       ['zfVel','Vertical velocity at upper surface','m/yr'], \
                       ['xbVel','Horizontal velocity x-component at lower surface','m/yr'], \
                       ['ybVel','Horizontal velocity y-component at lower surface','m/yr'], \
                       ['zbVel','Vertical velocity at lower surface','m/yr'], \
                       ['dragCoef','Basal friction coefficient','(units depend on law)'], \
                       ['viscosityCoef','viscosityCoef','-'], \
                       ['xxViscousTensor','xxViscousTensor','Pa'], \
                       ['yxViscousTensor','yxViscousTensor','Pa'], \
                       ['xyViscousTensor','xyViscousTensor','Pa'], \
                       ['yyViscousTensor','yyViscousTensor','Pa'], \
                       ['activeBasalThicknessSource','Mass balance at lower surface (active)','m/yr'], \
                       ['activeSurfaceThicknessSource','Mass balance at upper surface (active)','m/yr'], \
                       ['divergenceThicknessFlux','Divergence of horizontal flux','m/yr'], \
                       ['basalThicknessSource','Mass balance at lower surface','m/yr'], \
                       ['surfaceThicknessSource','Mass balance at upper surface','m/yr'], \
                       ['calvingFlux','Calving flux','-'] ])

      i = np.where(table[:,0] == vname)

      return table[i,1].item(), table[i,2].item()

    h5file = h5py.File(fname,'r')

    n_components = h5file.attrs['num_components']
    n_level = h5file.attrs['num_levels']

    levelsdx = []
    levelsboxes = []
    levelsx = []
    levelsy = []
    levelsdata = []

    for level in range(n_level):

      h5level = h5file['/level_'+str(level)+'/']

      dx = h5level.attrs['dx']
      # print(dx)

      h5box = h5level.get('boxes')
      h5box = np.array(h5box)
      h5data = h5level.get('data:datatype=0')
      h5data = np.array(h5data)
      h5offs = h5level.get('data:offsets=0')
      h5offs = np.array(h5offs)

      n_boxes = len(h5box)

      boxesx = []
      boxesy = []
      boxesdata = []

      for box in range(n_boxes):

        x = np.arange(h5box['lo_i'][box]-1,h5box['hi_i'][box]+2) * dx
        y = np.arange(h5box['lo_j'][box]-1,h5box['hi_j'][box]+2) * dx

        boxesx.append(x)
        boxesy.append(y)

        nx = len(x)
        ny = len(y)
        en = h5offs[box]

        # if level < 1:
        #  print(level, box, h5offs[box], h5box['lo_i'][box], h5box['hi_i'][box], h5box['lo_j'][box], h5box['hi_j'][box])

            # h5data[st:en] = np.where(h5data[st:en] == 0.0, np.NAN, h5data[st:en])

        st = en + tarcomponent * nx * ny
        en = en + (tarcomponent+1) * nx * ny

        boxesdata.append(h5data[st:en].reshape((ny,nx)))

      levelsx.append(boxesx)
      levelsy.append(boxesy)
      levelsdata.append(boxesdata)

      levelsdx.append(dx)
      levelsboxes.append(n_boxes)

    h5file.close

    self.fname = fname
    self.time = h5file.attrs['time']
    self.vname = h5file.attrs['component_'+str(tarcomponent)].decode('utf-8')
    self.fullname, self.units = variable_table(self.vname)
    self.levels = n_level
    self.boxes = levelsboxes
    self.dx = levelsdx
    self.x = levelsx
    self.y = levelsy
    self.data = levelsdata

    return None

  def plot(self, datamin=np.NAN, datamax=np.NAN):

    """
    thck.plot(datamin=0.0,datamax=100.0) both optional, defaults to min and max of data
    """

    def minmax(data):

      levmax = []
      levmin = []

      for level in range(len(data)):

        boxmax = []
        boxmin = []

        for box in range(len(data[level])):

          boxmax.append(np.amax(data[level][box]))
          boxmin.append(np.amin(data[level][box]))

        levmax.append(max(boxmax))
        levmin.append(min(boxmin))

      datamax = max(levmax)
      datamin = min(levmin)

      return datamin, datamax

    if np.isnan(datamin) | np.isnan(datamax):
      datamin, datamax = minmax(self.data)

    fig, ax = plt.subplots()

    color =['k', 'r', 'b', 'g', 'm', 'c']


    for level in range(self.levels):
      for box in range(self.boxes[level]):

        X, Y = np.meshgrid( \
               np.concatenate((self.x[level][box] - self.dx[level]/2.0, [self.x[level][box][-1] + self.dx[level]/2.0])), \
               np.concatenate((self.y[level][box] - self.dx[level]/2.0, [self.y[level][box][-1] + self.dx[level]/2.0])))

        # print(box, X.shape, self.x[level][box].shape)

        pcm = ax.pcolormesh(X, Y, self.data[level][box], shading = 'flat', cmap = 'hsv', \
                            vmin = datamin, vmax = datamax)
        # , edgecolor = 'k'
        rect = ptch.Rectangle((self.x[level][box][0:2].sum()/2.0, \
                               self.y[level][box][0:2].sum()/2.0), \
                               self.x[level][box][-2]-self.x[level][box][0], \
                               self.y[level][box][-2]-self.y[level][box][0], \
                               linewidth = 1, edgecolor = color[level], facecolor = 'none')

        ax.add_patch(rect)

    fig.colorbar(pcm,ax = ax, location = 'right', label = self.units)
    plt.title(self.fullname + ' in ' + self.units + ' at ' + '{:.2f}'.format(self.time) + ' years')

    ax.set_aspect('equal','box')

    fig.tight_layout()
    plt.show()

    return None

  def mesh(self):

    fig, ax = plt.subplots()

    marker = ['o', 's', 'D', 'x', '+', '.']
    color =['k', 'r', 'b', 'g', 'm', 'c']

    for level in range(self.levels):
      for box in range(self.boxes[level]):

        X, Y = np.meshgrid(self.x[level][box], self.y[level][box])

        # print(box, X.shape, self.x[level][box].shape)

        ax.scatter(X, Y, alpha = 0.5, marker = marker[level], c = color[level])

        rect = ptch.Rectangle((self.x[level][box][0:2].sum()/2.0, \
                               self.y[level][box][0:2].sum()/2.0), \
                               self.x[level][box][-2]-self.x[level][box][0], \
                               self.y[level][box][-2]-self.y[level][box][0], \
                               linewidth = 1, edgecolor = color[level], facecolor = 'none')
        ax.add_patch(rect)

    ax.set_aspect('equal','box')

    fig.tight_layout()
    plt.show()

    return None

def floatation(thick, bed, rhoi=917.0, rhoo=1023.6):

  float = deepcopy(thick)

  float.vname = 'float_thick'
  float.fullname = 'Thickness above flotation'
  float.units = 'm'

  print(rhoi, rhoo)

  for level in range(float.levels):
    for box in range(float.boxes[level]):

      float.data[level][box] = np.maximum(0.0,thick.data[level][box] + rhoo * np.minimum(0.0,bed.data[level][box]) / rhoi)
      float.data[level][box] = np.where(float.data[level][box] > 0.0, 1.0, 0.0)

  return float

def grounded(thck, bed, lsrf):

  def find_gl(d):

    imin = 0; imax = d.shape[0]-1
    jmin = 0; jmax = d.shape[1]-1

    for i in range(imin,imax):
      for j in range(jmin,jmax):

        #print(i, j)

        if ((d[i][j] == 2) | (d[i][j] == 4)) & \
            ((d[max(imin,i-1)][j] == 3) | (d[min(imax,i+1)][j] == 3) | \
            (d[i][min(jmax,j+1)] == 3) | (d[i][max(jmin,j-1)] == 3)):
          d[i][j] = 4

    return d

  ground = deepcopy(thck)

  ground.vname = 'mask'
  ground.fullname = 'Grounding mask'
  ground.units = '-'

  # look for differenecs greater than 1 mm
  criterion = 0.001

  for level in range(ground.levels):
    for box in range(ground.boxes[level]):

      # 0 open ocean
      # 1 ice-free land
      # 2 grounded ice
      # 3 floating ice
      # 5 grounding line

      conditions  = [(thck.data[level][box] == 0.0) & (bed.data[level][box] < 0.0), \
                     (thck.data[level][box] == 0.0) & (bed.data[level][box] >= 0.0), \
                     (thck.data[level][box] > 0.0) & (np.absolute(lsrf.data[level][box] - bed.data[level][box]) <= criterion), \
                     (thck.data[level][box] > 0.0) & (np.absolute(lsrf.data[level][box] - bed.data[level][box]) > criterion)] \

      ground.data[level][box] = np.select(conditions, [0, 1, 2, 3], default=-1)

      ground.data[level][box] = find_gl(ground.data[level][box])

  return ground

  # id grounding line points

def header(fname):

  # fname = 'plot.ase-mrf.4lev.000000.2d.hdf5'

  h5file = h5py.File(fname,'r')

  print('time (years) : '+str(h5file.attrs['time']))

  n_components = h5file.attrs['num_components']
  print('number of variables : '+str(n_components))

  for i in range(n_components):

    name = h5file.attrs['component_'+str(i)].decode('utf-8')

    print('variable '+str(i)+' : '+name)

  n_level = h5file.attrs['num_levels']
  print('number of levels : '+str(n_level))

  for i in range(n_level):

      dx = h5file['/level_'+str(i)+'/'].attrs['dx']
      print('dx (m) on level '+str(i)+' : '+str(dx))

  h5file.close

  return None




    # self.fname = 'none'#
    # self.time = 0#
    # self.vname = 'none'
    # self.fullname = 'none'
    # self.units = 'none'
    # self.levels = 0#
    # self.boxes = []#
    # self.dx = []#

    # self.data = []
    # for i in range(self.levels):
    #   self.data.append([None]*self.boxes[i])

    # self.x = self.data
    # self.y = self.data
