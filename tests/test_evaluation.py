import pytest
import os

import numpy as np
import pandas as pd
from devitoboundary.stencils.evaluation import (get_data_inc_reciprocals,
                                                split_types, add_distance_column,
                                                get_component_weights,
                                                find_boundary_points,
                                                apply_grid_offset, shift_grid_endpoint,
                                                fill_stencils, get_n_pts)
from devitoboundary.stencils.stencils import BoundaryConditions, StencilSet
from devito import Grid, Function, Dimension


class TestDistances:
    """
    Tests to verify the distances used in stencil evaluation.
    """
    def test_find_boundary_points(self):
        """Test that boundary points are correctly identified"""
        # Make some fake data
        data = np.full((5, 5, 5), -1)
        data[:, :, 2] = 0.5
        x, y, z = find_boundary_points(data)
        assert np.all(z == 2)
        assert x.size == 25
        assert y.size == 25

    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('spacing', [0.1, 1, 10])
    @pytest.mark.parametrize('offsets', [(-0.5, 0.), (0., 0.), (0.5, 0.),
                                         (-0.5, 0.5), (0.5, -0.5)])
    def test_reciprocal_calculation(self, axis, spacing, offsets):
        """
        A test to check that reciprocal eta calculated are consistent.
        """
        grid_offset = offsets[0]
        eval_offset = offsets[1]
        xyz = ('x', 'y', 'z')
        left = np.full((10, 10, 10), -2*spacing, dtype=float)
        right = np.full((10, 10, 10), -2*spacing, dtype=float)

        left_index = [None, None, None]
        left_index[axis] = 4
        right_index = [None, None, None]
        right_index[axis] = 5

        left[left_index[0], left_index[1], left_index[2]] = 0.3*spacing

        right[right_index[0], right_index[1], right_index[2]] = -0.7*spacing

        # Should produce the same results
        data_l = get_data_inc_reciprocals(left, spacing, xyz[axis], grid_offset, eval_offset)
        data_r = get_data_inc_reciprocals(right, spacing, xyz[axis], grid_offset, eval_offset)

        assert(np.all(np.isclose(data_l, data_r, equal_nan=True)))

    @pytest.mark.parametrize('offset', [-0.5, 0.5])
    def test_grid_shift(self, offset):
        """
        A test to check that grid offset is applied correctly to eta values
        """
        n_pts = 11
        x = 2*np.arange(2*n_pts)
        y = np.full(2*n_pts, 5)
        z = np.full(2*n_pts, 5)

        if offset == 0.5:
            eta_r = np.append(np.linspace(0, 0.9, n_pts), np.full(n_pts, np.NaN))
            eta_l = np.append(np.full(n_pts, np.NaN), np.linspace(0, -0.9, n_pts))
        elif offset == -0.5:
            eta_r = np.append(np.full(n_pts, np.NaN), np.linspace(0, 0.9, n_pts))
            eta_l = np.append(np.linspace(0, -0.9, n_pts), np.full(n_pts, np.NaN))
        else:
            raise ValueError("Invalid offset")

        frame = {'x': x, 'y': y, 'z': z, 'eta_l': eta_l, 'eta_r': eta_r}

        points = pd.DataFrame(frame)

        apply_grid_offset(points, 'x', offset, 0)

        assert np.all(points.x.to_numpy()[:17] == 2*np.arange(17))
        assert np.all(points.x.to_numpy()[17:] == 34 + 2*np.arange(5) - np.sign(offset))

        if offset == 0.5:
            assert np.all(np.logical_or(np.logical_and(points.eta_l <= 0, points.eta_l > -1),
                                        points.eta_r < 0.5))
        elif offset == -0.5:
            assert np.all(np.logical_or(np.logical_and(points.eta_r >= 0, points.eta_r < 1),
                                        points.eta_l > -0.5))

    @pytest.mark.parametrize('grid_offset', [-0.5, 0.5])
    def test_offset_skipping(self, grid_offset):
        """
        Test to check that eta manipulations are skipped when applying grid offset
        for points where both grid offset and evaluation offset are non-zero
        """
        n_pts = 11
        x = 2*np.arange(2*n_pts)
        y = np.full(2*n_pts, 5)
        z = np.full(2*n_pts, 5)

        if grid_offset == 0.5:
            eta_r = np.append(np.linspace(0, 0.9, n_pts), np.full(n_pts, np.NaN))
            eta_l = np.append(np.full(n_pts, np.NaN), np.linspace(0, -0.9, n_pts))
        elif grid_offset == -0.5:
            eta_r = np.append(np.full(n_pts, np.NaN), np.linspace(0, 0.9, n_pts))
            eta_l = np.append(np.linspace(0, -0.9, n_pts), np.full(n_pts, np.NaN))
        else:
            raise ValueError("Invalid offset")

        frame = {'x': x, 'y': y, 'z': z, 'eta_l': eta_l, 'eta_r': eta_r}

        points = pd.DataFrame(frame)

        offset_points = apply_grid_offset(points, 'x', grid_offset, -grid_offset)

        if grid_offset == -0.5:
            # Check left side
            assert np.all(np.isnan(offset_points.eta_l[11:]))
            assert np.all(offset_points.eta_l[:11] == np.linspace(0.5, -0.4, n_pts))
            # Check right side
            assert np.all(np.isnan(offset_points.eta_r[:11]))
            assert np.all(offset_points.eta_r[11:] == np.linspace(0.5, 1.4, n_pts))
        elif grid_offset == 0.5:
            # Check left side
            assert np.all(np.isnan(offset_points.eta_l[:11]))
            assert np.all(offset_points.eta_l[11:] == np.linspace(-0.5, -1.4, n_pts))
            # Check right side
            assert np.all(np.isnan(offset_points.eta_r[11:]))
            assert np.all(offset_points.eta_r[:11] == np.linspace(-0.5, 0.4, n_pts))

    @pytest.mark.parametrize('axis', [0, 1, 2])
    def test_type_splitting(self, axis):
        """
        A test to check that splitting of points into various categories
        functions as intended.
        """
        xyz = ('x', 'y', 'z')
        distances = np.full((10, 10, 10), -2, dtype=float)
        ind = [slice(None), slice(None), slice(None)]
        ind[axis] = np.array([1, 2, 5])
        distances[ind[0], ind[1], ind[2]] = 0.6

        # TODO: Ideally want to vary grid offset too
        data = get_data_inc_reciprocals(distances, 1, xyz[axis], 0, 0)
        add_distance_column(data)

        first, last, double, paired_left, paired_right = split_types(data,
                                                                     xyz[axis],
                                                                     10)

        assert(np.all(first.index.get_level_values(xyz[axis]).to_numpy() == 1))
        assert(np.all(last.index.get_level_values(xyz[axis]).to_numpy() == 6))
        assert(np.all(double.index.get_level_values(xyz[axis]).to_numpy() == 2))
        assert(np.all(paired_left.index.get_level_values(xyz[axis]).to_numpy() == 3))
        assert(np.all(paired_right.index.get_level_values(xyz[axis]).to_numpy() == 5))

    @pytest.mark.parametrize('offsets', [(-0.5, 0.), (0., 0.), (0.5, 0.),
                                         (-0.5, 0.5), (0.5, -0.5), (0., 0.5),
                                         (0., -0.5)])
    def test_shift_grid_endpoint(self, offsets):
        """
        Check that the endpoint shift to grab points where the grid node is on the
        exterior, but the stagger point is on the interior functions as intended.
        """
        grid_offset, eval_offset = offsets
        # shift_grid_endpoint(df, axis, grid_offset, eval_offset)

        # Coordinates for the points
        x = np.arange(10)
        y = np.arange(10)
        z = np.arange(10)

        if grid_offset == 0 and eval_offset == 0:
            # No shift needs applying here so skip
            pass
        else:
            # Shift will need applying for some points
            pass

        # Need to create a set of points where the shift needs applying
        # Need to create a set of points where no shift needs applying
        if grid_offset == 0:
            if eval_offset == 0.5:
                # eta l > -0.5 for no shift
                eta_l = np.linspace(-0.6, -0.9, 10)
                eta_r = np.NaN
                no_eta_l = np.linspace(-0.1, -0.4, 10)
                no_eta_r = np.NaN
            elif eval_offset == -0.5:
                # eta r < 0.5 for no shift
                eta_l = np.NaN
                eta_r = np.linspace(0.6, 0.9, 10)
                no_eta_l = np.NaN
                no_eta_r = np.linspace(0.1, 0.4, 10)
            else:
                # Set eta_l and eta_r to whatever
                eta_l = np.linspace(-0.1, -0.9, 10)
                eta_r = np.linspace(0.1, 0.9, 10)
                no_eta_l = np.linspace(-0.1, -0.9, 10)
                no_eta_r = np.linspace(0.1, 0.9, 10)
        elif grid_offset == -0.5:
            # eta r < 1 for no shift
            eta_l = np.NaN
            eta_r = np.linspace(1.1, 1.4, 10)
            no_eta_l = np.NaN
            no_eta_r = np.linspace(0.1, 0.9, 10)
        elif grid_offset == 0.5:
            # eta l > -1 for no shift
            eta_l = np.linspace(-1.1, -1.4, 10)
            eta_r = np.NaN
            no_eta_l = np.linspace(-0.1, -0.9, 10)
            no_eta_r = np.NaN

        # Dataframe of points which don't want shifting
        no_shift = pd.DataFrame({'x': x, 'y': y, 'z': z,
                                 'eta_l': no_eta_l, 'eta_r': no_eta_r})
        no_shift = no_shift.groupby(['z', 'y', 'x']).agg({'eta_l': 'min', 'eta_r': 'min'})
        no_shift['dist'] = 2

        no_shift_shifted = shift_grid_endpoint(no_shift, 'x', grid_offset, eval_offset)

        assert np.all(no_shift.index.get_level_values('x').to_numpy() == no_shift_shifted.index.get_level_values('x').to_numpy())
        if ~np.any(np.isnan(no_eta_l)):
            assert np.all(no_shift.eta_l.to_numpy() == no_shift_shifted.eta_l.to_numpy())
        if ~np.any(np.isnan(no_eta_r)):
            assert np.all(no_shift.eta_r.to_numpy() == no_shift_shifted.eta_r.to_numpy())
        assert np.all(no_shift.dist.to_numpy() == no_shift_shifted.dist.to_numpy())

        if grid_offset != 0 or eval_offset != 0:
            # Skip over case where no shift occurs
            # Dataframe of points which want shifting
            shift = pd.DataFrame({'x': x, 'y': y, 'z': z,
                                  'eta_l': eta_l, 'eta_r': eta_r})
            shift = shift.groupby(['z', 'y', 'x']).agg({'eta_l': 'min', 'eta_r': 'min'})
            shift['dist'] = 2

            shift_shifted = shift_grid_endpoint(shift, 'x', grid_offset, eval_offset)

            if grid_offset == 0:
                if eval_offset == 0.5:
                    inc = -1
                if eval_offset == -0.5:
                    inc = 1
            elif grid_offset == -0.5:
                inc = 1
            elif grid_offset == 0.5:
                inc = -1

            assert np.all(shift.index.get_level_values('x').to_numpy() + inc == shift_shifted.index.get_level_values('x').to_numpy())
            if ~np.any(np.isnan(eta_l)):
                assert np.all(shift.eta_l.to_numpy() - inc == shift_shifted.eta_l.to_numpy())
            if ~np.any(np.isnan(eta_r)):
                assert np.all(shift.eta_r.to_numpy() - inc == shift_shifted.eta_r.to_numpy())
            assert np.all(shift.dist.to_numpy() - inc == shift_shifted.dist.to_numpy())


class TestStencils:
    """
    A class containing tests to check stencil evaluation.
    """

    @pytest.mark.parametrize('offset', [0.5, -0.5])
    @pytest.mark.parametrize('point_type', ['first', 'last'])
    @pytest.mark.parametrize('order', [4, 6])
    @pytest.mark.parametrize('spacing', [0.1, 1., 10.])
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')  # y dimension of 1
    def test_fill_stencils_offset(self, offset, point_type, order, spacing):
        """
        Check that offsetting the grid and boundary by the same amount results
        in identical stencils for both cases. This is checked on both sides of
        the boundary. Note that this tests a larger chunk of the stack than the
        previous test.
        """

        spec = {2*i: 0 for i in range(1+order//2)}
        bcs = BoundaryConditions(spec, order)
        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        stencils = StencilSet(2, 0, bcs, cache=cache)
        lambdas = stencils.lambdaify
        max_ext_points = stencils.max_ext_points

        distances = np.full((10, 1, 10), -2*order*spacing, dtype=float)
        distances[4, :, :] = np.linspace(0, 0.9*spacing, 10)

        offset_distances = np.full((10, 1, 10), -2*order*spacing, dtype=float)
        if offset == 0.5:
            # +ve stagger
            offset_distances[4, :, :5] = np.linspace(0.5*spacing, 0.9*spacing, 5)
            offset_distances[5, :, 5:] = np.linspace(0, 0.4*spacing, 5)
        else:
            # -ve stagger
            offset_distances[4, :, :] = np.linspace(-0.5*spacing, 0.4*spacing, 10)

        data = get_data_inc_reciprocals(distances, spacing, 'x', 0, 0)
        offset_data = get_data_inc_reciprocals(offset_distances, spacing, 'x', offset, 0)
        dmask = np.full(21, True, dtype=bool)
        dmask[1] = False
        data = data[dmask]
        offset_data = offset_data[dmask]
        add_distance_column(data)
        add_distance_column(offset_data)
        if point_type == 'first':
            data = data[::2]
            data.dist = -order//2
            offset_data = offset_data[::2]
            offset_data.dist = -order//2
            # No need to drop points or shift grid endpoint, as that is done here

        else:
            data = data[1::2]
            data.dist = order//2
            offset_data = offset_data[1::2]
            offset_data.dist = order//2
            # No need to drop points or shift grid endpoint, as that is done here

        # Set n_pts
        data['n_pts'] = order//2
        offset_data['n_pts'] = order//2

        grid = Grid(shape=(10, 1, 10), extent=(9*spacing, 0, 9*spacing))
        s_dim = Dimension(name='s')
        ncoeffs = order + 1

        w_shape = grid.shape + (ncoeffs,)
        w_dims = grid.dimensions + (s_dim,)

        w_normal = Function(name='w_n', dimensions=w_dims, shape=w_shape)
        w_offset = Function(name='w_o', dimensions=w_dims, shape=w_shape)

        fill_stencils(data, point_type, max_ext_points, lambdas, w_normal)
        fill_stencils(offset_data, point_type, max_ext_points, lambdas, w_offset)

        if point_type == 'first':
            assert np.all(np.isclose(w_normal.data[2:5], w_offset.data[2:5]))
        else:
            assert np.all(np.isclose(w_normal.data[5:7], w_offset.data[5:7]))

    @pytest.mark.parametrize('order', [4, 6])
    @pytest.mark.parametrize('spec', [{'bcs': 'even', 'deriv': 1, 'goffset': 0., 'eoffset': 0.},
                                      {'bcs': 'odd', 'deriv': 1, 'goffset': 0., 'eoffset': 0.},
                                      {'bcs': 'even', 'deriv': 2, 'goffset': 0., 'eoffset': 0.}])
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')  # y and z dimensions of 1
    def test_zero_handling(self, order, spec):
        """
        Check that stencils with distances of zero evaluate correctly.
        """
        # Unpack the spec
        bc_type = spec['bcs']
        deriv = spec['deriv']
        goffset = spec['goffset']
        eoffset = spec['eoffset']
        if bc_type == 'even':
            bcs = BoundaryConditions({2*i: 0 for i in range(1+order//2)}, order)
        else:
            bcs = BoundaryConditions({2*i + 1: 0 for i in range(1+order//2)}, order)

        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        stencils = StencilSet(deriv, eoffset, bcs, cache=cache)
        lambdas = stencils.lambdaify
        max_ext_points = stencils.max_ext_points

        distances = np.full((10, 1, 1), -2*order, dtype=float)
        if goffset == 0.5:
            distances[4, :, :] = 0.5
        else:
            distances[4, :, :] = 0

        data = get_data_inc_reciprocals(distances, 1, 'x', goffset, eoffset)
        add_distance_column(data)
        data = data.iloc[1:-1]
        data = get_n_pts(data, 'double', order, eoffset)

        grid = Grid(shape=(10, 1, 11), extent=(9, 0, 0))
        s_dim = Dimension(name='s')
        ncoeffs = 2*max_ext_points + 1

        w_shape = grid.shape + (ncoeffs,)
        w_dims = grid.dimensions + (s_dim,)

        w = Function(name='w', dimensions=w_dims, shape=w_shape)

        fill_stencils(data, 'double', max_ext_points, lambdas, w)

        # Derivative stencils should be zero if evaluation offset is zero
        assert(np.all(np.abs(w.data) < np.finfo(np.float).eps))

    @pytest.mark.parametrize('side', ['first', 'last'])
    @pytest.mark.parametrize('order', [4, 6])
    @pytest.mark.parametrize('spec', [{'bcs': 'even', 'deriv': 1, 'goffset': 0., 'eoffset': 0.5},
                                      {'bcs': 'odd', 'deriv': 1, 'goffset': 0.5, 'eoffset': -0.5}])
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')  # y and z dimensions of 1
    def test_zero_handling_staggered(self, side, order, spec):
        """
        Check that stencils with distances of zero evaluate correctly for staggered
        systems.
        """
        # Unpack the spec
        bc_type = spec['bcs']
        deriv = spec['deriv']
        goffset = spec['goffset']
        eoffset = spec['eoffset']
        if bc_type == 'even':
            bcs = BoundaryConditions({2*i: 0 for i in range(1+order//2)}, order)
        else:
            bcs = BoundaryConditions({2*i + 1: 0 for i in range(1+order//2)}, order)

        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        # stencils_lambda = get_stencils_lambda(deriv, eoffset, bcs, cache=cache)
        stencils = StencilSet(deriv, eoffset, bcs, cache=cache)
        lambdas = stencils.lambdaify
        max_ext_points = stencils.max_ext_points

        distances = np.full((10, 1, 1), -2*order, dtype=float)
        if goffset == 0.5:
            distances[4, :, :] = 0.5
        else:
            distances[4, :, :] = 0

        data = get_data_inc_reciprocals(distances, 1, 'x', goffset, eoffset)
        add_distance_column(data)

        right_dist = pd.notna(data.eta_l)
        left_dist = pd.notna(data.eta_r)
        data.loc[right_dist, 'dist'] = order
        data.loc[left_dist, 'dist'] = -order

        first = data.loc[left_dist]
        last = data.loc[right_dist]
        first = shift_grid_endpoint(first, 'x', goffset, eoffset)
        last = shift_grid_endpoint(last, 'x', goffset, eoffset)
        first = get_n_pts(first, 'first', order, eoffset)
        last = get_n_pts(last, 'last', order, eoffset)

        grid = Grid(shape=(10, 1, 1), extent=(9, 0, 0))
        s_dim = Dimension(name='s')
        ncoeffs = 2*max_ext_points + 1

        w_shape = grid.shape + (ncoeffs,)
        w_dims = grid.dimensions + (s_dim,)

        w = Function(name='w', dimensions=w_dims, shape=w_shape)

        if side == 'first':
            fill_stencils(first, 'first', max_ext_points, lambdas, w)
        if side == 'last':
            fill_stencils(last, 'last', max_ext_points, lambdas, w)

        # Happy up to here, just need to figure out how to rewrite the last bit
        # Just check that the side of the stencil to one side of the center
        # is zero
        # Not ideal, but floating point error makes checking the exact values
        # a nightmare due to point selection issues (can go one way or the other)
        # However, either choice is valid.
        if side == 'first':
            for i in range(order//2):
                assert np.all(np.absolute(w.data[4-i, 0, 0, i-max_ext_points:]) < np.finfo(float).eps)
            # assert np.all(np.absolute(w.data[4-order//2:4]) < np.finfo(float).eps)
        else:  # Side is 'last'
            for i in range(order//2):
                assert np.all(np.absolute(w.data[4+i, 0, 0, :max_ext_points-i]) < np.finfo(float).eps)

    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('deriv', [1, 2])
    def test_get_component_weights(self, axis, deriv):
        """
        Check that get_component_weights returns stencils evaluated to their correct
        values.
        """
        def evaluate_variant(stencils_lambda, left_var, right_var,
                             left_eta, right_eta):
            """Evaluate the specified stencil"""
            stencil = stencils_lambda[left_var, right_var]
            eval_stencil = np.array([stencil[i](left_eta, right_eta)
                                     for i in range(5)])
            return eval_stencil

        def check_row(data, index, stencils_lambda):
            """Check values in the specified row are as intended"""
            indices = [slice(None), slice(None), slice(None)]
            indices[axis] = index
            stencils = data[indices[0], indices[1], indices[2]]

            if index == 0:
                true_stencil = evaluate_variant(stencils_lambda,
                                                0, 1, 0, 1.6)
            elif index == 1:
                true_stencil = evaluate_variant(stencils_lambda,
                                                0, 3, 0, 0.6)
            elif index == 2:
                true_stencil = evaluate_variant(stencils_lambda,
                                                4, 3, -0.4, 0.6)
            elif index == 3:
                true_stencil = evaluate_variant(stencils_lambda,
                                                4, 0, -0.4, 0)
            elif index == 4:
                true_stencil = evaluate_variant(stencils_lambda,
                                                2, 1, -1.4, 1.6)
            elif index == 5:
                true_stencil = evaluate_variant(stencils_lambda,
                                                0, 3, 0, 0.6)
            elif index == 6:
                true_stencil = evaluate_variant(stencils_lambda,
                                                4, 0, -0.4, 0)
            elif index == 7:
                true_stencil = evaluate_variant(stencils_lambda,
                                                2, 0, -1.4, 0)
            elif index >= 8:
                true_stencil = evaluate_variant(stencils_lambda,
                                                0, 0, 0, 0)

            misfit = stencils - true_stencil
            assert np.amax(np.absolute(misfit)) < 1e-6

        order = 4
        spec = {2*i: 0 for i in range(1+order//2)}
        bcs = BoundaryConditions(spec, order)

        grid = Grid(shape=(10, 10, 10), extent=(9., 9., 9.))
        function = Function(name='function', grid=grid, space_order=order)

        distances = np.full((10, 10, 10), -2, dtype=float)
        ind = [slice(None), slice(None), slice(None)]
        ind[axis] = np.array([1, 2, 5])
        distances[ind[0], ind[1], ind[2]] = 0.6

        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        stencils_lambda = get_stencils_lambda(deriv, 0, bcs, cache=cache)

        w = get_component_weights(distances, axis, function, deriv, stencils_lambda, 0)

        for i in range(10):
            check_row(w.data, i, stencils_lambda)

    @pytest.mark.parametrize('order', [4, 6])
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')  # y dimension of 1
    def test_special_case_stencils(self, order):
        """
        Check that special-case stencils required for staggered systems of equations
        are selected and evaluated correctly.
        """
        deriv = 1

        def evaluate_variant(stencils_lambda, left_var, right_var,
                             left_eta, right_eta):
            """Evaluate the specified stencil"""
            stencil = stencils_lambda[left_var, right_var]
            eval_stencil = np.array([stencil[i](left_eta, right_eta)
                                     for i in range(order+1)])
            return eval_stencil

        def check_stencil(data, stencils_lambda):
            """Check that the stencils produced are correct"""
            # Loop over indices of interest
            for x_ind in range(4-order//2, 5+order//2):
                if x_ind < 5:
                    left_var = 0
                    right_var = order + 1 - 2*(4-x_ind)
                    eta_l = np.zeros(10)
                    eta_r = np.linspace(-0.4, -0.1, 10) + (4-x_ind)
                else:
                    left_var = order+1 - 2*(x_ind-4)
                    right_var = 0
                    eta_l = np.linspace(-0.4, -0.1, 10) - (x_ind-4)
                    eta_r = np.zeros(10)
                for z_ind in range(10):
                    true_stencil = evaluate_variant(stencils_lambda, left_var, right_var,
                                                    eta_l[z_ind], eta_r[z_ind])

                    misfit = data[x_ind, :, z_ind] - true_stencil
                    assert np.amax(np.absolute(misfit)) < 1e-6

        bcs = BoundaryConditions({2*i + 1: 0 for i in range(1+order//2)}, order)

        grid = Grid(shape=(10, 1, 10), extent=(9., 0., 9.))
        x, y, z = grid.dimensions
        function = Function(name='function', grid=grid, space_order=order, staggered=x)

        distances = np.full((10, 1, 10), -order//2, dtype=float)
        distances[4, :] = np.linspace(0.1, 0.4, 10)[np.newaxis, :]

        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        stencils_lambda = get_stencils_lambda(deriv, -0.5, bcs, cache=cache)

        w = get_component_weights(distances, 0, function, deriv, stencils_lambda, -0.5)

        check_stencil(w.data, stencils_lambda)
