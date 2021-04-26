"""
Functions used in the evaluation of modified stencils for implementing
immersed boundaries.
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from devito import Coefficient, Dimension, Function
from devitoboundary import __file__
from devitoboundary.stencils.stencils import get_stencils_lambda
from devitoboundary.stencils.stencil_utils import standard_stencil, get_grid_offset


def find_boundary_points(data):
    """
    Find points immediately adjacent to boundary and return their locations.

    Parameters
    ----------
    data : ndarray
        The distance function for a particular axis

    Returns
    -------
    x, y, z : ndarray
        The x, y, and z indices of boundary adjacent points (where "boundary
        adjacent" refers to points where the distance function is not at its
        filler values).
    """
    fill_val = np.amin(data)  # Filler value for the distance function
    x, y, z = np.where(data != fill_val)

    return x, y, z


def build_dataframe(data, spacing):
    """
    Return a dataframe with columns x, y, z, eta

    Parameters
    ----------
    data : ndarray
        The distance function for a particular axis
    spacing : float
        The spacing of the grid

    Returns
    -------
    points : pandas DataFrame
        The dataframe of points and their respective eta values
    """
    x, y, z = find_boundary_points(data)

    eta = data[x, y, z]

    col_x = pd.Series(x.astype(np.int), name='x')
    col_y = pd.Series(y.astype(np.int), name='y')
    col_z = pd.Series(z.astype(np.int), name='z')

    eta_r = pd.Series(np.where(eta >= 0, eta/spacing, np.NaN), name='eta_r')
    eta_l = pd.Series(np.where(eta <= 0, eta/spacing, np.NaN), name='eta_1')

    frame = {'x': col_x, 'y': col_y, 'z': col_z, 'eta_l': eta_l, 'eta_r': eta_r}
    points = pd.DataFrame(frame)
    return points


def apply_grid_offset(df, axis, offset):
    """
    Shift eta values according to grid offset. Carried out in place
    """
    df.eta_l -= offset
    df.eta_r -= offset

    if np.sign(offset) == 1:
        eta_r_mask = df.eta_r < 0
        eta_l_mask = df.eta_l <= -1

        df.loc[eta_r_mask, 'eta_l'] = df.eta_r[eta_r_mask]
        df.loc[eta_r_mask, 'eta_r'] = np.NaN

        df.loc[eta_l_mask, 'eta_l'] += 1

        df.loc[eta_l_mask, axis] -= 1

    elif np.sign(offset) == -1:
        eta_r_mask = df.eta_r >= 1
        eta_l_mask = df.eta_l > 0

        df.loc[eta_l_mask, 'eta_r'] = df.eta_l[eta_l_mask]
        df.loc[eta_l_mask, 'eta_l'] = np.NaN

        df.loc[eta_r_mask, 'eta_r'] -= 1

        df.loc[eta_r_mask, axis] += 1

    df = df.agg({'eta_l': 'max', 'eta_r': 'min'})


def calculate_reciprocals(df, axis, side):
    """
    Calculate reciprocal distances from known values.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of points for which reciprocals should be calculated
    axis : str
        The axis along which reciprocals should be calculated. Should be 'x',
        'y', or 'z'
    side : str
        The side from which reciprocals should be calculated

    Returns
    -------
    reciprocals : pandas DataFrame
        The dataframe of reciprocal points and their eta values
    """
    increment = -1 if side == 'l' else 1
    other = 'eta_r' if side == 'l' else 'eta_l'  # Other side
    column = 'eta_' + side

    reciprocals = df[pd.notna(df[column])].copy()
    reciprocals[axis] += increment
    reciprocals[other] = reciprocals[column] - increment
    reciprocals[column] = np.NaN

    return reciprocals


def get_data_inc_reciprocals(data, spacing, axis, offset):
    """
    Calculate and consolidate reciprocal values, returning resultant dataframe.

    Parameters
    ----------
    data : ndarray
        The axial distance function data for the specified axis
    spacing : float
        The grid spacing for the specified axis
    axis : str
        The specified axis
    offset : float
        The grid offset for this axis

    Returns
    -------
    aggregated_data : pandas DataFrame
        Dataframe of points including reciprocal distances, consolidated down
    """

    df = build_dataframe(data, spacing)

    apply_grid_offset(df, axis, offset)

    reciprocals_l = calculate_reciprocals(df, axis, 'l')
    reciprocals_r = calculate_reciprocals(df, axis, 'r')

    full_df = df.append([reciprocals_l, reciprocals_r])

    # Group and aggregate to consolidate points doubled up by this process
    aggregated_data = full_df.groupby(['z', 'y', 'x']).agg({'eta_l': 'min', 'eta_r': 'min'})

    return aggregated_data


def add_distance_column(data):
    """Adds a distance column to the dataframe and initialises with zeroes"""
    data['dist'] = 0


def split_types(data, axis, axis_size):
    """
    Splits points into the five categories:

    first        o->|
    last         |<-o
    double       |<-o->|
    paired_left  |<-o--------->|
    paired_right |<---------o->|

    Where o marks point location and | is the boundary

    Parameters
    ----------
    data : pandas DataFrame
        The dataframe containing the points to be split
    axis : str
        The axis for which points should be split
    axis_size : int
        The length of the axis (in terms of grid points)

    Returns
    -------
    first, last, double, paired_left, paired_right : pandas DataFrame
        Dataframes for each of the categories
    """
    levels = ['z', 'y', 'x']
    levels.remove(axis)

    # FIXME: Can copies be removed?
    first = data.groupby(level=levels).head(1).copy()  # Get head
    last = data.groupby(level=levels).tail(1).copy()  # Get tail

    double_cond = np.logical_and(pd.notna(data.eta_l),
                                 pd.notna(data.eta_r))
    double = data[double_cond]  # Get double-sided

    f_index = first.index  # Get the indices of the easy categories
    l_index = last.index
    d_index = double.index

    paired = data.drop(f_index).drop(l_index).drop(d_index)  # Remove these bits
    paired_left = paired[pd.notna(paired.eta_l)].copy()
    paired_right = paired[pd.notna(paired.eta_r)].copy()

    # Paired points in paired_left and paired_right match up
    paired_dist = (paired_right.index.get_level_values(axis).to_numpy()
                   - paired_left.index.get_level_values(axis).to_numpy())
    paired_left.dist = paired_dist
    paired_right.dist = -paired_dist

    paired_left.eta_r = paired_right.eta_r.to_numpy()  # Ugly, since in most cases these wouldn't fit together
    paired_right.eta_l = paired_left.eta_l.to_numpy()

    first.dist = -f_index.get_level_values(axis).to_numpy()
    last.dist = axis_size - 1 - l_index.get_level_values(axis).to_numpy()

    # Drop any values outside computational domain (generated by reciprocity)
    # Needs to be done after the drop or invalid values end up in the paired categories
    first = first[first.index.get_level_values(axis) >= 0]
    last = last[last.index.get_level_values(axis) < axis_size]

    return first, last, double, paired_left, paired_right


def evaluate_stencils(df, point_type, n_stencils, left_variants, right_variants,
                      space_order, stencil_lambda, grid_offset):
    """
    Evaluate the stencils associated with a set of boundary-adjacent
    points.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of boundary-adjacent points
    point_type : string
        The category of the points. Can be 'first', 'last', 'double',
        'paired_left', or 'paired_right'.
    n_stencils : int
        The number of stencils associated with each point. Each point must have
        the same number of stencils associated with it.
    left_variants: ndarray
        The left-side stencil variants for each of the stencils
    right_variants: ndarray
        The right-side stencil variants for each of the stencils
    space_order : int
        The space order of the function for which stencils are to be evaluated
    stencil_lambda : ndarray
        The functions for stencils to be evaluated
    grid_offset : float
        The function offset along the supplied axis

    Returns
    -------
    stencils : ndarray
        The evaluated stencil coefficients
    """
    # The base "index" for eta
    eta_base = np.tile(np.arange(n_stencils), (left_variants.shape[0], 1))
    # Initialise empty stencil array
    stencils = np.zeros(left_variants.shape + (space_order + 1,), dtype=np.float32)

    if point_type == 'first':
        eta_right = np.tile(df.eta_r.to_numpy()[:, np.newaxis],
                            (1, n_stencils)) + n_stencils - eta_base - 1

        for right_var in range(space_order+1):
            mask = right_variants == right_var
            for coeff in range(space_order+1):
                func = stencil_lambda[0, right_var, coeff]
                stencils[mask, coeff] = func(0, eta_right[mask] - grid_offset)

    if point_type == 'last':
        eta_left = np.tile(df.eta_l.to_numpy()[:, np.newaxis],
                           (1, n_stencils)) - eta_base

        for left_var in range(space_order+1):
            mask = left_variants == left_var
            for coeff in range(space_order+1):
                func = stencil_lambda[left_var, 0, coeff]
                stencils[mask, coeff] = func(eta_left[mask] - grid_offset, 0)

    if point_type == 'double':
        eta_left = df.eta_l.to_numpy()[:, np.newaxis]
        eta_right = df.eta_r.to_numpy()[:, np.newaxis]

        for left_var in range(space_order+1):
            for right_var in range(space_order+1):
                mask = np.logical_and(left_variants == left_var,
                                      right_variants == right_var)
                for coeff in range(space_order+1):
                    func = stencil_lambda[left_var, right_var, coeff]
                    stencils[mask, coeff] = func(eta_left[mask] - grid_offset,
                                                 eta_right[mask] - grid_offset)

    if point_type == 'paired_left':
        dst = df.dist.to_numpy()[:, np.newaxis]
        eta_left = np.tile(df.eta_l.to_numpy()[:, np.newaxis],
                           (1, n_stencils)) - eta_base
        eta_right = np.tile(df.eta_r.to_numpy()[:, np.newaxis],
                            (1, n_stencils)) + dst - eta_base

        for left_var in range(space_order+1):
            for right_var in range(space_order+1):
                mask = np.logical_and(left_variants == left_var,
                                      right_variants == right_var)
                for coeff in range(space_order+1):
                    func = stencil_lambda[left_var, right_var, coeff]
                    stencils[mask, coeff] = func(eta_left[mask] - grid_offset,
                                                 eta_right[mask] - grid_offset)

    if point_type == 'paired_right':
        dst = df.dist.to_numpy()[:, np.newaxis]
        # dst used to have a minus (this was wrong)
        eta_left = np.tile(df.eta_l.to_numpy()[:, np.newaxis],
                           (1, n_stencils)) + n_stencils + dst - eta_base - 1
        eta_right = np.tile(df.eta_r.to_numpy()[:, np.newaxis],
                            (1, n_stencils)) + n_stencils - eta_base - 1

        for left_var in range(space_order+1):
            for right_var in range(space_order+1):
                mask = np.logical_and(left_variants == left_var,
                                      right_variants == right_var)
                for coeff in range(space_order+1):
                    func = stencil_lambda[left_var, right_var, coeff]
                    stencils[mask, coeff] = func(eta_left[mask] - grid_offset,
                                                 eta_right[mask] - grid_offset)
    # print("Stencils "+point_type)
    # print(stencils)
    return stencils


def fill_weights(points, stencils, point_type, weights, axis, n_pts=1):
    """
    Fill a specified Function with the corresponding weights.

    Parameters:
    points : pandas DataFrame
        The set of boundary-adjacent points from which the function should
        be filled.
    stencils : ndarray
        The set of stencils to fill with
    point_type : string
        The category of the points. Can be 'first', 'last', 'double',
        'paired_left', or 'paired_right'.
    weights : devito Function
        The Function to fill with stencil coefficients
    axis : str
        The axis along which the stencils are orientated. Can be 'x', 'y', or
        'z'.
    n_pts : int
        The number of stencils associated with each point. Each point must have
        the same number of stencils associated with it.
    """
    x = points.index.get_level_values('x').values
    y = points.index.get_level_values('y').values
    z = points.index.get_level_values('z').values

    if point_type == 'first' or point_type == 'paired_right':
        if axis == 'x':
            for i in range(n_pts):
                weights.data[x-n_pts+1+i, y, z] = stencils[:, i, :]
        elif axis == 'y':
            for i in range(n_pts):
                weights.data[x, y-n_pts+1+i, z] = stencils[:, i, :]
        elif axis == 'z':
            for i in range(n_pts):
                weights.data[x, y, z-n_pts+1+i] = stencils[:, i, :]

    elif point_type == 'last' or point_type == 'paired_left':
        if axis == 'x':
            for i in range(n_pts):
                weights.data[x+i, y, z] = stencils[:, i, :]
        elif axis == 'y':
            for i in range(n_pts):
                weights.data[x, y+i, z] = stencils[:, i, :]
        elif axis == 'z':
            for i in range(n_pts):
                weights.data[x, y, z+i] = stencils[:, i, :]

    elif point_type == 'double':
        weights.data[x, y, z] = stencils[:, 0, :]


def get_variants(df, space_order, point_type, axis, stencils, weights,
                 grid_offset, eval_offset):
    """
    Get the all the stencil variants associated with the points, evaluate them,
    and fill the respective positions in the weight function.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of boundary-adjacent points
    space_order : int
        The order of the function for which stencils are to be generated
    point_type : string
        The category of the points. Can be 'first', 'last', 'double',
        'paired_left', or 'paired_right'.
    axis : str
        The axis along which the stencils are orientated. Can be 'x', 'y', or
        'z'.
    stencils : ndarray
        The functions for stencils to be evaluated
    weights : devito Function
        The Function to fill with stencil coefficients
    grid_offset : float
        The function offset along the supplied axis
    eval_offset : float
        The relative offset at which derivatives should be evaluated
    """
    if point_type == 'first':
        n_pts = np.minimum(int(space_order/2), 1-df.dist.to_numpy())
        # Modifier for points which lie within half a grid spacing of the boundary
        modifier_right = np.where(df.eta_r.to_numpy() - grid_offset < 0.5, 0, 1)

        # It is necessary to account for staggering during stencil selection
        # stagger_mod_r = np.where(df.eta_r.to_numpy() - grid_offset >= grid_offset + eval_offset, 0, 1)
        stagger_mod_r = 0

        # Starting point for the right stencil (moving from left to right)
        start_right = space_order-2*(n_pts-1)-modifier_right+stagger_mod_r

        i_min = np.amin(n_pts)
        i_max = np.amax(n_pts)

        for i in np.linspace(i_min, i_max, 1+i_max-i_min, dtype=int):
            mask = n_pts == i
            mask_size = np.count_nonzero(mask)
            left_variants = np.zeros((mask_size, i), dtype=int)

            # This is capped at space_order to prevent invalid variant numbers
            right_variants = np.minimum(np.tile(2*np.arange(i), (mask_size, 1))
                                        + start_right[mask, np.newaxis],
                                        space_order)

            # Iterate over left and right variants
            eval_stencils = evaluate_stencils(df[mask], 'first', i,
                                              left_variants, right_variants,
                                              space_order, stencils,
                                              grid_offset)

            # Insert the stencils into the weight function
            fill_weights(df[mask], eval_stencils, 'first',
                         weights, axis, n_pts=i)

    elif point_type == 'last':
        n_pts = np.minimum(int(space_order/2), 1+df.dist.to_numpy())
        # Modifier for points which lie within half a grid spacing of the boundary
        modifier_left = np.where(df.eta_l.to_numpy() - grid_offset > -0.5, 0, 1)

        # It is necessary to account for staggering during stencil selection
        # stagger_mod_l = np.where(df.eta_l.to_numpy() - grid_offset <= grid_offset + eval_offset, 0, 1)
        stagger_mod_l = 0

        start_left = space_order-modifier_left+stagger_mod_l

        i_min = np.amin(n_pts)
        i_max = np.amax(n_pts)
        for i in np.linspace(i_min, i_max, 1+i_max-i_min, dtype=int):
            mask = n_pts == i
            mask_size = np.count_nonzero(mask)
            # This is capped at space_order to prevent invalid variant numbers
            left_variants = np.minimum(np.tile(-2*np.arange(i), (mask_size, 1))
                                       + start_left[mask, np.newaxis],
                                       space_order)

            right_variants = np.zeros((mask_size, i), dtype=int)

            # Iterate over left and right variants
            eval_stencils = evaluate_stencils(df[mask], 'last', i,
                                              left_variants, right_variants,
                                              space_order, stencils,
                                              grid_offset)

            # Insert the stencils into the weight function
            fill_weights(df[mask], eval_stencils, 'last',
                         weights, axis, n_pts=i)

    elif point_type == 'double':
        n_pts = 1
        # Modifier for points which lie within half a grid spacing of the boundary
        modifier_left = np.where(df.eta_l.to_numpy() - grid_offset > -0.5, 0, 1)
        modifier_right = np.where(df.eta_r.to_numpy() - grid_offset < 0.5, 0, 1)

        # It is necessary to account for staggering during stencil selection
        # stagger_mod_l = np.where(df.eta_l.to_numpy() - grid_offset <= grid_offset + eval_offset, 0, 1)
        # stagger_mod_r = np.where(df.eta_r.to_numpy() - grid_offset >= grid_offset + eval_offset, 0, 1)
        stagger_mod_l = 0
        stagger_mod_r = 0

        start_left = space_order-modifier_left+stagger_mod_l
        start_right = space_order-modifier_right+stagger_mod_r

        # This is capped at space_order to prevent invalid variant numbers
        left_variants = np.minimum(start_left[:, np.newaxis], space_order)
        right_variants = np.minimum(start_right[:, np.newaxis], space_order)

        # Iterate over left and right variants
        eval_stencils = evaluate_stencils(df, 'double', 1,
                                          left_variants, right_variants,
                                          space_order, stencils,
                                          grid_offset)

        # Insert the stencils into the weight function
        fill_weights(df, eval_stencils, 'double', weights, axis)

    elif point_type == 'paired_left':
        n_pts = np.minimum(int(space_order/2), df.dist.to_numpy())
        # Modifier for points which lie within half a grid spacing of the boundary
        modifier_left = np.where(df.eta_l.to_numpy() - grid_offset > -0.5, 0, 1)
        modifier_right = np.where(df.eta_r.to_numpy() - grid_offset < 0.5, 0, 1)

        # It is necessary to account for staggering during stencil selection
        # stagger_mod_l = np.where(df.eta_l.to_numpy() - grid_offset <= grid_offset + eval_offset, 0, 1)
        # stagger_mod_r = np.where(df.eta_r.to_numpy() - grid_offset >= grid_offset + eval_offset, 0, 1)
        stagger_mod_l = 0
        stagger_mod_r = 0

        start_left = space_order-modifier_left+stagger_mod_l
        start_right = space_order-2*df.dist.to_numpy()-modifier_right+stagger_mod_r

        i_min = np.amin(n_pts)
        i_max = np.amax(n_pts)
        for i in np.linspace(i_min, i_max, 1+i_max-i_min, dtype=int):
            mask = n_pts == i
            mask_size = np.count_nonzero(mask)
            # This is capped at space_order to prevent invalid variant numbers
            left_variants = np.minimum(np.tile(-2*np.arange(i), (mask_size, 1))
                                       + start_left[mask, np.newaxis],
                                       space_order)
            right_variants = np.minimum(np.maximum(np.tile(2*np.arange(i), (mask_size, 1))
                                                   + start_right[mask, np.newaxis], 0),
                                        space_order)

            # Iterate over left and right variants
            eval_stencils = evaluate_stencils(df[mask], 'paired_left', i,
                                              left_variants, right_variants,
                                              space_order, stencils,
                                              grid_offset)
            # Insert the stencils into the weight function
            fill_weights(df[mask], eval_stencils, 'paired_left',
                         weights, axis, n_pts=i)

    elif point_type == 'paired_right':
        n_pts = np.minimum(int(space_order/2),
                           1-df.dist.to_numpy()-np.minimum(int(space_order/2),
                                                           -df.dist.to_numpy()))
        # Modifier for points which lie within half a grid spacing of the boundary
        modifier_left = np.where(df.eta_l.to_numpy() - grid_offset > -0.5, 0, 1)
        modifier_right = np.where(df.eta_r.to_numpy() - grid_offset < 0.5, 0, 1)

        # It is necessary to account for staggering during stencil selection
        # stagger_mod_l = np.where(df.eta_l.to_numpy() - grid_offset <= grid_offset + eval_offset, 0, 1)
        # stagger_mod_r = np.where(df.eta_r.to_numpy() - grid_offset >= grid_offset + eval_offset, 0, 1)
        stagger_mod_l = 0
        stagger_mod_r = 0

        start_left = space_order+2*df.dist.to_numpy()-modifier_left+stagger_mod_l
        start_right = space_order-2*(n_pts-1)-modifier_right+stagger_mod_r

        i_min = np.amin(n_pts)
        i_max = np.amax(n_pts)
        for i in np.linspace(i_min, i_max, 1+i_max-i_min, dtype=int):
            mask = n_pts == i
            mask_size = np.count_nonzero(mask)
            # This is capped at space_order to prevent invalid variant numbers
            left_variants = np.minimum(np.maximum(np.tile(-2*np.arange(i), (mask_size, 1))
                                                  + start_left[mask, np.newaxis], 0),
                                       space_order)
            right_variants = np.minimum(np.tile(2*np.arange(i), (mask_size, 1))
                                        + start_right[mask, np.newaxis],
                                        space_order)

            # Iterate over left and right variants
            eval_stencils = evaluate_stencils(df[mask], 'paired_right', i,
                                              left_variants, right_variants,
                                              space_order, stencils,
                                              grid_offset)

            # Insert the stencils into the weight function
            fill_weights(df[mask], eval_stencils, 'paired_right',
                         weights, axis, n_pts=i)


def get_component_weights(data, axis, function, deriv, stencils, offset):
    """
    Take a component of the distance field and return the associated weight
    function.

    Parameters
    ----------
    data : ndarray
        The field of the axial distance function for the specified axis
    axis : int
        The axis along which the stencils are orientated. Can be 0, 1, or 2
    function : devito Function
        The function for which stencils should be calculated
    deriv : int
        The order of the derivative to which the stencils pertain
    stencils : ndarray
        The functions for stencils to be evaluated
    offset : float
        The offset at which the default stencil should be evaluated

    Returns
    -------
    w : devito Function
        Function containing the stencil coefficients
    """
    grid_offset = get_grid_offset(function, axis)
    """
    print("Axis", axis)
    print("Function space dimensions", function.space_dimensions)
    print("Grid offset", grid_offset)
    print("Function stagger", function.staggered)
    """

    f_grid = function.grid
    axis_dim = 'x' if axis == 0 else 'y' if axis == 1 else 'z'

    full_data = get_data_inc_reciprocals(data, f_grid.spacing[axis], axis_dim, grid_offset)

    add_distance_column(full_data)

    first, last, double, paired_left, paired_right \
        = split_types(full_data, axis_dim, f_grid.shape[axis])

    # Additional dimension for storing weights
    s_dim = Dimension(name='s')
    ncoeffs = function.space_order + 1

    w_shape = f_grid.shape + (ncoeffs,)
    w_dims = f_grid.dimensions + (s_dim,)

    w = Function(name='w_'+function.name+'_'+axis_dim, dimensions=w_dims, shape=w_shape)

    w.data[:] = standard_stencil(deriv, function.space_order, offset=offset)

    # Fill the stencils
    get_variants(first, function.space_order, 'first',
                 axis_dim, stencils, w, grid_offset, offset)
    get_variants(last, function.space_order, 'last',
                 axis_dim, stencils, w, grid_offset, offset)

    # Check lengths before doing these three
    if len(double.index) != 0:
        get_variants(double, function.space_order, 'double',
                     axis_dim, stencils, w, grid_offset, offset)
    if len(paired_left.index) != 0:
        get_variants(paired_left, function.space_order, 'paired_left',
                     axis_dim, stencils, w, grid_offset, offset)
    if len(paired_right.index) != 0:
        get_variants(paired_right, function.space_order, 'paired_right',
                     axis_dim, stencils, w, grid_offset, offset)
    """
    for i in range(function.space_order+1):
        plt.imshow(w.data[:, 50, :, i].T)
        plt.title(function.name + " " + axis_dim + " coeff " + str(i))
        plt.colorbar()
        plt.show()
    """
    w.data[:] /= f_grid.spacing[axis]**deriv  # Divide everything through by spacing

    return w


def get_weights(data, function, deriv, bcs, offsets=(0, 0, 0)):
    """
    Get the modified stencil weights for a function and derivative given the
    axial distances.

    Parameters
    ----------
    data : devito VectorFunction
        The axial distance function
    function : devito Function
        The function for which stencils should be calculated
    deriv : int
        The order of the derivative to which the stencils pertain
    bcs : list of devito Eq
        The boundary conditions which should hold at the surface
    offsets : tuple of int
        The offset at which the function should be evaluated for each axis.
        Default is (0, 0, 0).

    Returns
    -------
    substitutions : Devito Substitutions
        The substitutions to be included in the devito equation
    """
    cache = os.path.dirname(__file__) + '/extrapolation_cache.dat'

    # This wants to start as an empty list
    weights = []
    for axis in range(3):
        # Check any != filler value in data[axis].data
        # FIXME: Could just calculate this rather than finding the minimum
        fill_val = np.amin(data[axis].data)
        if np.any(data[axis].data != fill_val):
            # If True, then behave as normal
            # If False then pass
            stencils = get_stencils_lambda(deriv, offsets[axis], bcs, cache=cache)

            axis_weights = get_component_weights(data[axis].data, axis, function,
                                                 deriv, stencils, offsets[axis])
            print(deriv, function, function.grid.dimensions[axis], axis_weights)
            # this should be an append
            weights.append(Coefficient(deriv, function,
                                       function.grid.dimensions[axis],
                                       axis_weights))
        else:
            pass  # No boundary-adjacent points so don't return any subs
    # Raise error if list is empty
    if len(weights) == 0:
        raise ValueError("No boundary-adjacent points in provided fields")
    return tuple(weights)
