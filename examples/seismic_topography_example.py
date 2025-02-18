import numpy as np
import pandas as pd

from devito import Grid, TimeFunction, Eq, solve, Operator, ConditionalDimension
from devitoboundary import ImmersedBoundary, BoundaryConditions
from examples.seismic import TimeAxis, RickerSource

# Parameters
qc = False
toggle_normals = False

C = 0.1  # Courant number
VP = 1.2  # P wave velocity

# Grid configuration
# 10.8 x 10.8 x 5.4 km
# 50m spacing
extent = (10800., 10800., 5400.)
shape = (217, 217, 109)  # (109, 109, 55)
origin = (0., 0., -3900.)
grid = Grid(shape=shape, extent=extent, origin=origin)

# Time series configuration
t0 = 0.  # Simulation starts at t=0
tn = 4000.  # Simulation length in ms
dt = C*grid.spacing[0]/(VP)

steps = int((t0+tn)/dt)+2

# Configure the source
time_range = TimeAxis(start=t0, stop=tn, step=dt)
f0 = 0.002  # 2Hz
src = RickerSource(name='src', grid=grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in x and y dimensions, then set depth
src.coordinates.data[0, :-1] = 5400.  # Centered
src.coordinates.data[0, -1] = -500  # 500m below sea level

# Set up snapshotting
nsnaps = 100  # Want 100 snapshots
factor = round(steps / nsnaps)

time_subsampled = ConditionalDimension('t_sub', parent=grid.time_dim,
                                       factor=factor)
usave = TimeFunction(name='usave', grid=grid, time_order=2, space_order=2,
                     save=(steps + factor - 1) // factor,
                     time_dim=time_subsampled)

u = TimeFunction(name='u', grid=grid,
                 space_order=4, time_order=2,
                 coefficients='symbolic')

# Surface configuration
infile = 'topography/crater_lake.ply'
# Zero even derivatives on the boundary
spec = {2*i: 0 for i in range(u.space_order)}
bcs_u = BoundaryConditions(spec, u.space_order)
functions = pd.DataFrame({'function': [u],
                          'bcs': [bcs_u]},
                         columns=['function', 'bcs'])

# Create the immersed boundary surface
surface = ImmersedBoundary('topography', infile, functions,
                           interior_point=tuple(src.coordinates.data[0]),
                           qc=qc, toggle_normals=toggle_normals)

# Configure derivative needed
derivs = pd.DataFrame({'function': [u],
                       'derivative': [2],
                       'eval_offset': [(0., 0., 0.)]},
                      columns=['function', 'derivative', 'eval_offset'])
coeffs = surface.subs(derivs)

# We can now write the PDE
pde = VP*u.dt2 - u.laplace
eq = Eq(pde, 0, coefficients=coeffs)

# And set up the update
stencil = solve(eq.evaluate, u.forward)

# Our injection term
src_term = src.inject(field=u.forward, expr=src*dt**2/VP)

# Now create our operator
op = Operator([Eq(u.forward, stencil)] + [Eq(usave, u)] + src_term)

# And run
op.apply(dt=dt)

outfile = 'data/seismic_topography_wavefield.npy'
np.save(outfile, usave.data)
"""
plot_extent = [0, grid.extent[0],
               origin[2], grid.extent[2] + origin[2]]
for i in range(usave.data.shape[0] - 1):
    fig = plt.figure()
    plt.imshow(usave.data[i, :, int(grid.shape[1]/2), :].T,
               origin='lower', extent=plot_extent,
               vmin=-6, vmax=6, cmap='seismic')
    plt.colorbar()
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.savefig("images/image-%s" % str(i))
    # plt.show()
    plt.close()
"""
