"""
A module for implementation of topography in Devito via use of
the immersed boundary method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ['Boundary']


class Boundary():
    """
    An object to contain data relevant for implementing the
    immersed boundary method on a given domain.
    """

    def __init__(self, grid, boundary_func,
                 inv_boundary_func=None,
                 method_order=4):
        self._method_order = method_order

        # Derive useful properties from grid (make these hidden)
        self._shape = np.asarray(grid.shape)
        self._extent = np.asarray(grid.extent)
        self._spacing = grid.spacing

        if not callable(boundary_func):
            raise NotImplementedError

        # Create primary node list
        self._primary_nodes(boundary_func)

        if 0 < self._shape.size < 3:
            # Create full node list
            self._node_list()
        else:
            raise NotImplementedError

        #self._eta_list(boundary_func, inv_boundary_func,
        #               shape, extent, spacing)

    @property
    def method_order(self):
        """
        Order of the FD discretisation.
        Currently this is only implemented for 4th order stencils.
        """
        return self._method_order

    def _primary_nodes(self, boundary_func):
        """
        Compute the primary boundary nodes, from which other nodes in the
        boundary sub-domain can be derived.
        """

        if self._shape.size > 2:
            raise NotImplementedError

        if self._shape.size == 1:
            x_coords = np.linspace(0, self._extent[0], self._shape[0])
            # In this case the boundary is a single node
            boundary = boundary_func()
            p_nodes = np.array(np.floor(boundary/self._spacing[0]).astype(int))
            # These two cases shouldn't occur in 1D:
            if p_nodes < 0 or p_nodes >= self._shape[0]:
                raise ValueError("Given boundary location is not \
                                  in the computational domain.")
        elif self._shape.size == 2:
            x_coords = np.linspace(0, self._extent[0], self._shape[0])
            boundary = boundary_func(x_coords)
            p_nodes = np.floor(boundary/self._spacing[1]).astype(int)
            # Represent nodes outside the computational domain with -1.
            p_nodes[p_nodes < 0] = -1
            p_nodes[p_nodes >= self._shape[1]] = -1
        else:
            # FIX ME: Add 3D case etc.
            raise NotImplementedError

        self._primary_nodes = p_nodes

        return self._primary_nodes

    def _node_list(self):
        """
        Generates a list of possible nodes (with redundancy)
        that require their stencil to be modified.
        """

        p_nodes = self._primary_nodes


        # Default box size around node
        def_size = max(np.array([self.method_order/2-1, 1], dtype=int))

        # Check if we're dealing with a 1D case
        if self._shape.size == 1:
            node_dict = ()
            for i in range(def_size+1):
                node_dict += (p_nodes-def_size+i,)
            self._node_list = node_dict
            return self._node_list
        # Check if we're dealing with 2D case
        # Forgive me
        elif self._shape.size == 2:
            node_dict = ()
            for i in range(p_nodes.size):
                down_pos = p_nodes[i]
                while down_pos >= 0:
                    node_dict += ((i, down_pos),)
                    for j in range(def_size): # Project def_size out
                        if i-def_size+j >= 0:
                            if down_pos <= min(p_nodes[i-def_size+j:i+1]):
                                # Project left
                                node_dict += ((i-def_size+j, down_pos),)
                        if i+def_size-j <= p_nodes.size-1:
                            if down_pos <= min(p_nodes[i:i+def_size-j+1]):
                                # Project right
                                node_dict += ((i+def_size-j, down_pos),)
                    if i <= p_nodes.size-2:
                        if p_nodes[i+1]+1 < down_pos:
                            down_pos -= 1
                        elif i >= 0:
                            if p_nodes[i-1]+1 < down_pos:
                                down_pos -= 1
                            else:
                                break
                    elif i >= 0:
                        if p_nodes[i-1]+1 < down_pos:
                            down_pos -= 1
                        else:
                            break

                for j in range(def_size+1): # Project def_size out
                    if p_nodes[i]-def_size+j >= 0:
                        # Project down
                        node_dict += ((i, p_nodes[i]-def_size+j),)

            # Remove repeated entries
            node_dict = tuple(set(node_dict))
            # Some dud points may appear with particularly weird boundaries
            # Difficult to efficiently eliminate due to sparsity

            self._node_list = node_dict

            return self._node_list

    def show_nodes(self, savefig=False, savepath=None):
        """
        Produces a plot of all nodes at which the function will be modified.
        """
        assert self._shape.size != 1, "Nodes cannot be plotted in 1D"

        if self._shape.size == 2:
            node_list = self._node_list
            x_coords = np.linspace(0, self._extent[0], self._shape[0])
            y_coords = np.linspace(0, self._extent[1], self._shape[1])
            x_points = []
            y_points = []
            for pos in node_list:
                x_points.append(x_coords[pos[0]])
                y_points.append(y_coords[pos[1]])
            plt.plot(x_points, y_points, 'rx')
            plt.xlim(0, self._extent[0])
            plt.ylim(0, self._extent[1])
            plt.xlabel("x")
            plt.ylabel("y")
            # Allow user to save figure
            if savefig:
                if savepath is not None:
                    plt.savefig(str(savepath), dpi=200)
                else:
                    raise FileNotFoundError('No filepath specifed for saving.')
            plt.show()

#    def _eta_list(self, boundary_func, inv_boundary_func,
#                  shape, extent, spacing):
#        """
#        Generates relevant values of eta for all nodes where stencils are
#        to be modified.
#        """
#
#        p_nodes = self._primary_nodes
#        node_list = self._node_list
#
#        x_coords = np.linspace(0, extent[0], shape[0])
#        # eta in the positive x direction
#        xr_list = ()
#
#        # Check if we're dealing with a 1D case
#        if shape.size == 1:
#            for node in node_list:
#                xr_list += ((boundary_func() - x_coords[node])/spacing[0],)
#        # Check if we're dealing with a 2D case
#        elif shape.size == 2:
#            y_coords = np.linspace(0, extent[1], shape[1])
#            y_list = ()
#            # In 2D, there is also an eta in the negative x direction
#            xl_list = ()
#
#            for node in node_list:
#                # For y direction
#                y_list += ((boundary_func(x_coords[node[0]])
#                            - y_coords[node[1]])/spacing[1],)
#                # For xr and xl directions
#                # Use inverse for now (will use search eventually)
