import sympy as sp
import numpy as np
import pickle

from devito.logger import warning
from devitoboundary.stencils.stencil_utils import standard_stencil
from devitoboundary.symbolics.symbols import (a, x_b, x_a, x_t, E, eta_l, eta_r)

__all__ = ['taylor', 'BoundaryConditions', 'get_ext_coeffs', 'get_stencils', 'get_stencils_lambda']


def taylor(x, order):
    """Generate a taylor series expansion of a given order"""
    n = sp.symbols('n')
    polynomial = sp.Sum(a[n]*(x-x_b)**n, (n, 0, order)).doit()
    return polynomial


class BoundaryConditions:
    """
    Contains information on a given set of boundary conditions.

    Parameters
    ----------
    bcs : dict
        A dict of {derivative: value}
    order : int
        The order of the finite difference discretization

    Methods
    -------
    get_taylor: Get the taylor series with appropriate coefficient modifications

    Attributes
    ----------
    bcs: The bcs contained by this object

    order: The order of the discretization these bcs are associated with

    x: The variable used for the taylor series

    """
    def __init__(self, bcs, order):
        self._bcs = bcs
        self._order = order

        self._x = sp.symbols('x')

    def __str__(self):
        return "BoundaryConditions(bcs: {}, order: {})".format(self._bcs, self._order)

    def get_taylor(self, order=None):
        """Get the taylor series with appropriate coefficient modifications"""
        if order is None:
            order = self.order
        series = taylor(self._x, order)
        for deriv, val in self._bcs.items():
            series = series.subs(a[deriv], val/np.math.factorial(deriv))
        return series

    @property
    def bcs(self):
        """The bcs contained by this object"""
        return self._bcs

    @property
    def order(self):
        """The order of the discretization these bcs are associated with"""
        return self._order

    @property
    def x(self):
        """The variable used for the taylor series"""
        return self._x


def get_ext_coeffs(bcs, cache=None):
    """
    Get the extrapolation coefficients for a set of boundary conditions

    Parameters
    ----------
    bcs : BoundaryConditions
        The boundary conditions to be applied
    cache : str
        The path to the extrapolation cache. Optional.

    Returns
    -------
    coeff_dict : dict
        The dictionary of extrapolation coefficients
    """
    if cache is None:
        return _get_ext_coeffs(bcs)
    else:
        try:
            with open(cache, 'rb') as f:
                coeff_cache = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Invalid cache location")

        if not isinstance(coeff_cache, dict):
            raise TypeError("Specified file does not contain a dictionary")

        # Unique key for this extrapolation
        key = str(bcs)

        try:
            coeff_dict = coeff_cache[key]
        except KeyError:
            warning("Extrapolation not in cache. Generating new extrapolation")
            coeff_dict = _get_ext_coeffs(bcs)

            # Add new entry to dictionary
            coeff_cache[key] = coeff_dict
            # And update the cache
            with open(cache, 'wb') as f:
                pickle.dump(coeff_cache, f)

        return coeff_dict


def _get_ext_coeffs(bcs):
    """Get the extrapolation coefficients for a set of boundary conditions"""
    n_pts = bcs.order//2  # Number of interior points
    coeff_dict = {}  # Master coefficient dictionary
    for points_count in range(1, n_pts+1):
        # This -1 might want to be taken into account somewhere else
        taylor = bcs.get_taylor(order=2*points_count - 1)
        lhs = sum([E[point]*taylor.subs(bcs.x, x_a[point]) for point in range(points_count)])
        rhs = taylor.subs(bcs.x, x_t)
        eqs = [sp.Eq(sp.expand(lhs).coeff(a[i], 1), sp.expand(rhs).coeff(a[i], 1)) for i in range(bcs.order+1)
               if sp.Eq(lhs.coeff(a[i], 1), rhs.coeff(a[i], 1)) is not sp.true]

        # Variables to solve for
        solve_vars = [E[point] for point in range(points_count)]
        coeffs = sp.solve(eqs, solve_vars)

        coeff_dict[points_count] = coeffs

    return coeff_dict


def get_stencils(deriv, offset, bcs, cache=None):
    """
    Get the array of stencils for a given specification

    Parameters:
    deriv : int
        The derivative for which stencils should be generated
    offset : float
        The offset at which the derivative should be evaluated
    bcs : BoundaryConditions
        The boundary conditions which these stencils are for
    cache : str
        Path to the extrapolation cache. Optional

    Returns
    -------
    stencil_array : ndarray
        The array of stencils

    """
    def get_unusable(variant):
        """Get the number of unusable points on a side given the variant"""
        return min(variant, int(variant/2+1))

    def get_outside(variant):
        """Get the number of exterior points on a side given the variant"""
        return int(np.ceil(variant/2))

    def get_available(left_unusable, right_unusable, left_outside, right_outside, s_o):
        """
        Get the positions of points available for the extrapolation on each side
        given the number of unusable and exterior stencil points.
        """
        # Deal with -0 = 0 when indexing from right
        if right_unusable == 0:
            left_available = range(-s_o//2, s_o//2+1)[left_unusable:]
            right_available = range(-s_o//2, s_o//2+1)[left_outside:]
        else:
            left_available = range(-s_o//2, s_o//2+1)[left_unusable:-right_outside]
            right_available = range(-s_o//2, s_o//2+1)[left_outside:-right_unusable]
        return left_available, right_available

    def get_stencil_addition(outside, available, coeff_dict, s_o, base_stencil, side):
        """
        Get the additions to the base stencil introduced by the extrapolations for a side
        """
        if side != 'left' and side != 'right':
            raise ValueError("Invalid side")

        n_coeffs = s_o//2 if len(available) > s_o//2 else len(available) if len(available) > 0 else 1

        if side == 'left':
            # Force use of middle point if nomianally no points are available
            ext_points = tuple(available[:s_o]) if len(available) > 0 else (outside-s_o//2,)

            # Build main substitutions
            main_subs = [(x_a[i], ext_points[i]) for i in range(n_coeffs)]

            # Set floor on eta if necessary
            if len(available) > 0:
                main_subs += [(x_b, eta_l)]
            else:
                main_subs += [(x_b, -0.5)]
        else:
            # Force use of middle point if nomianally no points are available
            ext_points = tuple(available[-s_o:]) if len(available) > 0 else (s_o//2-outside,)

            # Build main substitutions (Index from right to left for this one)
            main_subs = [(x_a[i], ext_points[-1-i]) for i in range(n_coeffs)]

            # Set floor on eta if necessary
            if len(available) > 0:
                main_subs += [(x_b, eta_r)]
            else:
                main_subs += [(x_b, 0.5)]

        # Apply main substitutions
        ext_coeffs = coeff_dict[n_coeffs].copy()
        ext_coeffs = {coeff: val.subs(main_subs) for (coeff, val) in ext_coeffs.items()}

        points = tuple(range(-s_o//2, s_o//2+1))

        # Array containing additions to stencil
        additions = np.full(s_o+1, sp.Float(0))

        if side == 'left':
            for point in range(outside):
                # Apply substitutions for x_t
                point_coeffs = {coeff: val.subs(x_t, points[point])
                                for (coeff, val) in ext_coeffs.items()}

                # Loop over the coefficients and add them to the addition with the correct weighting
                for position in range(n_coeffs):
                    additions[s_o//2 + ext_points[position]] += base_stencil[point]*point_coeffs[E[position]]
        else:
            for point in range(outside):
                # Apply substitutions for x_t
                point_coeffs = {coeff: val.subs(x_t, points[-1-point])
                                for (coeff, val) in ext_coeffs.items()}

                # Loop over the coefficients and add them to the addition with the correct weighting
                for position in range(n_coeffs):
                    additions[s_o//2 + ext_points[-1-position]] += base_stencil[-1-point]*point_coeffs[E[position]]

        return additions

    def get_stencil_additions(left_outside, right_outside,
                              left_available, right_available,
                              coeff_dict, s_o, base_stencil):
        """
        Get the additions to the base stencil introduced by the extrapolations
        """
        left_add = get_stencil_addition(left_outside, left_available, coeff_dict,
                                        s_o, base_stencil, 'left')
        right_add = get_stencil_addition(right_outside, right_available, coeff_dict,
                                         s_o, base_stencil, 'right')

        stencil = base_stencil.copy()

        stencil += left_add + right_add
        stencil[:left_outside] = sp.Float(0)
        if right_outside != 0:
            stencil[-right_outside:] = sp.Float(0)

        return stencil

    s_o = bcs.order
    base_stencil = standard_stencil(deriv, s_o,
                                    offset=offset, as_float=False)

    # If True, then an extra variant should be generated on that side
    # This variant will be used where the offset point and respective index
    # point lie on opposite sides of a boundary
    extra_right = False
    extra_left = False
    if np.sign(offset) == -1:
        # Negative evaluation offset, may need another right variant
        extra_right = True
    elif np.sign(offset) == 1:
        # Positive evaluation offset, may need another left variant
        extra_left = True

    # Extend range for non-zero offsets
    stencil_array = np.empty((s_o+1+extra_left, s_o+1+extra_right, s_o+1), dtype=object)

    coeff_dict = get_ext_coeffs(bcs, cache=cache)

    # Loop over variants
    # Extend range for non-zero offsets
    for left in range(s_o + 1 + extra_left):
        left_unusable = get_unusable(left)
        left_outside = get_outside(left)
        for right in range(s_o + 1 + extra_right):
            right_unusable = get_unusable(right)
            right_outside = get_outside(right)

            left_available, right_available = get_available(left_unusable, right_unusable,
                                                            left_outside, right_outside, s_o)

            stencil = get_stencil_additions(left_outside, right_outside,
                                            left_available, right_available,
                                            coeff_dict, s_o, base_stencil)
            stencil_array[left, right] = stencil

    return stencil_array


def get_stencils_lambda(deriv, offset, bcs, cache=None):
    """
    Get the stencils as an array of functions which can be called on supplied values
    of eta_l and eta_r.

    Parameters:
    deriv : int
        The derivative for which stencils should be generated
    offset : float
        The offset at which the derivative should be evaluated
    bcs : BoundaryConditions
        The boundary conditions which these stencils are for
    cache : str
        Path to the extrapolation cache. Optional

    Returns
    -------
    funcs : ndarray
        The array of functions for each stencil coefficient. Indexed by [left variant, right variant, index].

    """
    stencils = get_stencils(deriv, offset, bcs, cache=cache)
    funcs = np.empty(stencils.shape, dtype=object)
    for i in range(stencils.size):
        funcs.flat[i] = sp.lambdify([eta_l, eta_r], stencils.flat[i])
    return funcs
