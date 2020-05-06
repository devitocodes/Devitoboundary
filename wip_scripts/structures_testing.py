from devito import Grid
from devitoboundary import PolyMesh
import numpy as np
np.seterr(all='raise')

data = [[1, 2, 3], [2, 1, 0], [5, 1, 2],
        [5, 3, 1], [1, 3, 5], [1, 2, 8],
        [2, 5, 7], [4, 1, 2], [8, 5, 2],
        [17, 15, 1], [13, 15, 2], [13, 14, 3],
        [1.3, 3.5, 2], [4.8, 2, 4], [2.6, 3, 2],
        [0, 0, 1], [0, 15, 1], [17, 0, 1]]

grid = Grid(extent=(20, 20, 20), shape=(11, 11, 11))

mesh = PolyMesh(data, grid)

# Will also want to figure out what to do if a point lies on a surface
q_points = [[4.2, 2.2, 5.3], [3.1, 1.1, 5.6], [1.4, 6.2, 1.2],
            [10.0, 3.2, 1.1], [6.3, 2.2, 6.1], [7.2, 2.1, 4.2]]

# mesh.query(q_points)
