# python_reader
Some python code to read and display the hdf5 files used by BISICLES.

How to use

import bisiclesh5 as b5

b5.header(filename)

provides a summary of the data in filename (e.g., 'plot.ase-mrf.4lev.001000.2d.hdf5') such as time, variables, level, boxes, cell sizes etc. 

thck =  b5.bisicles_var(filename,n)
  
where fname is the hdf5 that you want to read (e.g., 'plot.ase-mrf.4lev.001000.2d.hdf5') and n is an integer identifying the variable that you want, which you can get from the output of the reader (e.g., variable 0 : thickness).

The output 'thck' is an instance of the class bisicles_var which contains the variable on the system of grid meshes:

    thck.fname =  original file name
    thck.time = time in model years
    thck.vname = variable name in short form
    thck.fullname = variable name in long form
    thck.units = variable's units
    thck.levels = number of levels 
    thck.boxes = number of boxes in each level (a list, e.g., thck.boxes[0])
    thck.dx = cell size (dx) for each level in m (a list, e.g., thck.dx[0])
    thck.x = grid cell locations (centrepoints) in m (a 2d list of numpy arrays, e.g. thck.dx[0][0])
    thck.y = grid cell locations (centrepoints) in m (a 2d list of 1d numpy arrays, e.g. thck.dy[0][0])
    thck.data = the variable itself (a 2d list of dd numpy arrays, e.g. thck.data[0][0])
    
note that thck.x, thck.y and thck.data include a halo of ghost cells one cell wide on all four sides.

thck.plot()

a method to the data in thck.

thck.mesh()

a method to plot the grid cells locations in thck.

