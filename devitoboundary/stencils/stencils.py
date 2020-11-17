"""
A module for stencil generation given a set of boundary conditions, and a method
order.
"""
import numpy as np
import sympy as sp
import pickle
import warnings

from devito import Eq
from devitoboundary.symbolics.symbols import (x_a, u_x_a, n_max, a, x_b, x_l,
                                              x_r, x_c)


__all__ = ['StencilGen']


class StencilGen:
    """
    Stencil_Gen(space_order, staggered=False, stencil_file=None)

    Modified stencils for an immersed boundary at which a set of boundary conditions
    are to be imposed.

    Parameters
    ----------
    space_order : int
        The order of the desired spatial discretization.
    staggered : bool
        Switch to stagger stencils. Default is False.
    stencil_file : str
        The filepath of the stencil cache.

    Attributes
    ----------
    stencil_list : list
        A nested list of possible stencil variants. Indexed by [left variant],
        [right variant]. A variant is the number of half grid increments by
        which the stencil is truncated by the boundary.
    space_order : int
        The order of the stencils.
    x_b : Sympy symbol
        The generic boundary position to be used for specifying boundary
        conditions.

    Methods
    -------
    u(val, deriv=0)
        The generic function, to be used for specifying bounday conditions.
    add_bcs(bc_list)
        Add a list of boundary conditions constructed using u and x_b. Must be
        called before all_variants().
    all_variants(deriv)
        Calculate the stencil coefficients of all possible stencil variants
        required for a given derivative.
    """

    def __init__(self, s_o, staggered=False, stencil_file=None):
        self._s_o = s_o
        self._staggered = staggered

        self._x = sp.IndexedBase('x')  # Arbitrary values of x
        self._u_x = sp.IndexedBase('u_x')  # Respective values of the function

        self._a = sp.IndexedBase('a')
        self._n, self._n_max = sp.symbols('n, n_max')  # Maximum polynomial order

        self._x_b, self._x_r, self._x_l = sp.symbols('x_b, x_r, x_l')
        self._x_c = sp.symbols('x_c')  # Continuous x

        self._f = sp.IndexedBase('f')  # Function values at particular points
        self._h_x = sp.symbols('h_x')  # Grid spacing
        self._eta_l, self._eta_r = sp.symbols('eta_l, eta_r')  # Distance to boundary

        self._bcs = None
        self._stencil_list = None
        self._i_poly_variants = None
        self._u_poly_variants = None

        if stencil_file is None:
            self._stencil_dict = {}
        else:
            with open(stencil_file, 'rb') as f:
                stencils = pickle.load(f)
                if not isinstance(stencils, dict):
                    raise TypeError("Specified file does not contain a dictionary")
                self._stencil_dict = stencils
        self._stencil_file = stencil_file

    @property
    def bc_list(self):
        """The list of boundary conditions"""
        return self._bcs

    @property
    def stencil_list(self):
        """The list of all possible stencils"""
        return self._stencil_list

    @property
    def space_order(self):
        """The formal order of the stencils"""
        return self._s_o

    @property
    def x_b(self):
        """The generic boundary position"""
        return self._x_b

    def u(self, val, deriv=0):
        """
        Returns specified derivative of the extrapolations polynomial. To be used
        for specification of boundary conditions.

        Parameters
        ----------
        val : Sympy symbol
            The variable of the function. Should typically be x_b.
        deriv : int
            The order of the derivative. Default is zero.
        """
        x_poly = sp.symbols('x_poly')
        polynomial = sp.Sum(self._a[self._n]*x_poly**self._n,
                            (self._n, 0, self._n_max))
        return sp.diff(polynomial, x_poly, deriv).subs(x_poly, val)

    def add_bcs(self, bc_list):
        """
        Add a list of boundary conditions. These conditions should be formed as
        devito Eq objects equating some u(x_b) to a given value.

        Parameters
        ----------
        bc_list : list
            The list of boundary conditions.
        """
        self._bcs = bc_list

    def _coeff_gen(self, n_pts, bcs=None):
        """
        Generate extrapolation polynomial coefficients given the number of
        interior points available.
        """

        def point_count(n_bcs, n_pts):
            """
            The number of points used by the polynomial, given number of bcs
            and points available.
            """
            # Points to be used can be no larger than number available
            # Points required is equal to space_order - number of bcs
            # At least one point must be used
            return min(max(self._s_o - n_bcs + 1, 1), n_pts)

        def extr_poly_order(n_bcs, n_p_used):
            """
            The order of the polynomial required given number of boundary
            conditions and points to be used.
            """
            return n_bcs + n_p_used - 1

        def reduce_order(bcs, poly_order):
            """
            Return a reduction in polynomial order, since boundary conditions
            which evaluate to zero will result in polynomial coefficients which
            are functions of one another otherwise.
            """
            eval_bcs = [Eq(bcs[i].lhs.subs(n_max, poly_order).doit(),
                           bcs[i].rhs) for i in range(len(bcs))]
            return eval_bcs.count(Eq(0, 0))

        def evaluate_equations(equations, poly_order):
            """
            Evaluate the sums in the equation list to the specified order.
            """
            for i in range(len(equations)):
                equations[i] = Eq(equations[i].lhs.subs(n_max, poly_order).doit(),
                                  equations[i].rhs)

        def solve_for_coeffs(equations, poly_order):
            """
            Return the coefficients of the extrapolation polynomial
            """
            solve_variables = tuple(a[i] for i in range(poly_order+1))
            return sp.solve(equations, solve_variables)

        if bcs is None:
            bcs = self._bcs
        n_bcs = len(bcs)

        n_p_used = point_count(n_bcs, n_pts)

        poly_order = extr_poly_order(n_bcs, n_p_used)
        poly_order -= reduce_order(bcs, poly_order)

        # Generate additional equations for each point used
        eq_list = [Eq(self.u(x_a[i]), u_x_a[i]) for i in range(n_p_used)]

        equations = bcs + eq_list

        evaluate_equations(equations, poly_order)

        return solve_for_coeffs(equations, poly_order)

    def _poly_variants(self):
        """
        Generate all possible polynomial variants required given the order of the
        spatial discretization and a list of boundary conditions. There will be
        a single polynomial generated for the independent case, and
        one for each unified case as available points are depleted.
        """

        def double_sided_bcs(bcs):
            """Turn single sided set of bcs into double sided"""
            double_bcs = []
            for i in range(len(bcs)):
                double_bcs.append(self._bcs[i].subs(x_b, x_l))
                double_bcs.append(self._bcs[i].subs(x_b, x_r))
            return double_bcs

        def generate_double_sided(bcs):
            """
            Generate double sided polynomials based on boundary conditions
            imposed at both ends of a stencil.
            """
            ds_poly = []

            # Set up double-sided boundary conditions list
            ds_bcs = double_sided_bcs(bcs)

            # Unique extrapolation for each number of interior points available
            # Can't have less than one interior point
            # Maximum number of interior points required is one more than the
            # space order minus the number of boundary conditions
            n_bcs = len(self._bcs)
            for i in range(1, self._s_o - n_bcs + 1):
                ds_poly_coeffs = self._coeff_gen(self._s_o - n_bcs + 1 - i,
                                                 bcs=ds_bcs)

                ds_poly.append(sum([ds_poly_coeffs[a[j]]*x_c**j
                               for j in range(len(ds_poly_coeffs))]))

            return ds_poly

        def generate_single_sided(bcs):
            """
            Generate a single-sided polynomial based on boundary conditions
            imposed.
            """
            n_bcs = len(self._bcs)
            ss_poly_coeffs = self._coeff_gen(self._s_o - n_bcs + 1)
            ss_poly = [ss_poly_coeffs[a[i]]*x_c**i
                       for i in range(len(ss_poly_coeffs))]

            return ss_poly

        n_bcs = len(self._bcs)

        # Package into function generate_double_sided
        # Initialise list for storing polynomials
        ds_poly = []
        # Set up ds_bc_list
        ds_bc_list = []
        for i in range(n_bcs):
            ds_bc_list.append(self._bcs[i].subs(self._x_b, self._x_l))
            ds_bc_list.append(self._bcs[i].subs(self._x_b, self._x_r))

        for i in range(1, self._s_o - n_bcs + 1):
            ds_poly_coeffs = self._coeff_gen(self._s_o - n_bcs + 1 - i,
                                             bcs=ds_bc_list)
            ds_poly_i = 0
            for j in range(len(ds_poly_coeffs)):
                ds_poly_i += ds_poly_coeffs[self._a[j]]*self._x_c**j
            ds_poly.append(ds_poly_i)

        # Package into function generate_single_sided
        ss_poly_coeffs = self._coeff_gen(self._s_o - n_bcs + 1)
        ss_poly = 0
        for i in range(len(ss_poly_coeffs)):
            ss_poly += ss_poly_coeffs[self._a[i]]*self._x_c**i
        self._i_poly_variants = ss_poly
        self._u_poly_variants = ds_poly

    def all_variants(self, deriv, stencil_out=None):
        """
        Calculate the stencil coefficients of all possible stencil variants
        required for a given derivative.

        Parameters
        ----------
        deriv : int
            The derivative for which stencils should be calculated
        stencil_out : str
            The filepath to where the stencils should be cached. This will
            default to the filepath set at initialization. If this is not done,
            the filepath supplied here will be used. If both are missing,
            then stencils will not be cached.
        """

        try:
            self._stencil_list = self._stencil_dict[str(self._bcs)+str(self._s_o)+str(deriv)+'ns']
        except KeyError:
            if stencil_out is None and self._stencil_file is None:
                no_warn = "No file specified for caching generated stencils."
                warnings.warn(no_warn)
            if stencil_out is not None and self._stencil_file is not None:
                dupe_warn = "File already specified for caching stencils. Defaulting to {}"
                warnings.warn(dupe_warn.format(self._stencil_file))

            warnings.warn("Generating new stencils, this may take some time.")
            self._all_variants(deriv)

            if self._stencil_file is not None:
                with open(self._stencil_file, 'wb') as f:
                    pickle.dump(self._stencil_dict, f)
            elif stencil_out is not None:
                with open(stencil_out, 'wb') as f:
                    pickle.dump(self._stencil_dict, f)

    def _all_variants(self, deriv):
        """
        Calculate the stencil coefficients of all possible stencil variants
        required for a given derivative.

        Parameters
        ----------
        deriv : int
            The derivative for which stencils should be calculated
        """
        # Want to start by calculating standard stencil expansions
        base_coeffs = sp.finite_diff_weights(deriv,
                                             range(-int(self._s_o/2), int(self._s_o/2)+1),
                                             0)[-1][-1]
        base_stencil = 0
        for i in range(len(base_coeffs)):
            base_stencil += base_coeffs[i]*self._f[i-int(self._s_o/2)]

        # Get the polynomial variants
        self._poly_variants()

        # Set up empty nested list of size MxM
        self._stencil_list = [[None for i in range(self._s_o+1)] for j in range(self._s_o+1)]

        # Number of boundary conditions
        n_bcs = len(self._bcs)

        for le in range(self._s_o+1):
            # Left interval
            for ri in range(self._s_o+1):
                # Right interval
                # Set stencil [le, ri]
                if (le != 0 or ri != 0):
                    stencil_entry = base_stencil

                    # Points unusable on right
                    right_u = min(ri, int(ri/2+1))
                    # Points outside on right
                    right_o = int(np.ceil(ri/2))
                    # Points unusable on left
                    left_u = min(le, int(le/2+1))
                    # Points outside on left
                    left_o = int(np.ceil(le/2))
                    # Available points for right poly
                    a_p_right = self._s_o + 1 - right_u - left_o
                    # Available points for left poly
                    a_p_left = self._s_o + 1 - left_u - right_o
                    # Use a unified polynomial if less than s_o - n_bcs + 1 points available

                    # Points to use for right poly
                    u_p_right = min(self._s_o - n_bcs + 1, a_p_right)
                    # Points to use for left poly
                    u_p_left = min(self._s_o - n_bcs + 1, a_p_left)

                    if a_p_right >= self._s_o - n_bcs + 1 and a_p_left >= self._s_o - n_bcs + 1:
                        # Use separate polynomials
                        # Right side polynomial
                        if ri != 0:
                            r_poly = self._i_poly_variants

                            for n in range(u_p_right+1):
                                # Substitute in correct values of x and u_x
                                r_poly = r_poly.subs([(self._u_x[n], self._f[int(self._s_o/2)-right_u-n]),
                                                     (self._x[n], (int(self._s_o/2)-right_u-n)*self._h_x)])

                            # Also need to replace x_b with eta_r*h_x
                            r_poly = r_poly.subs(self._x_b, self._eta_r*self._h_x)

                            for n in range(right_o):
                                stencil_entry = stencil_entry.subs(self._f[int(self._s_o/2)-n],
                                                                   r_poly.subs(self._x_c, (int(self._s_o/2)-n)*self._h_x))
                        else:
                            r_poly = None

                        # Left side polynomial
                        if le != 0:
                            l_poly = self._i_poly_variants

                            for n in range(u_p_left+1):
                                # Substitute in correct values of x and u_x
                                l_poly = l_poly.subs([(self._u_x[n], self._f[n+left_u-int(self._s_o/2)]),
                                                     (self._x[n], (n+left_u-int(self._s_o/2))*self._h_x)])

                            # Also need to replace x_b with eta_l*h_x
                            l_poly = l_poly.subs(self._x_b, self._eta_l*self._h_x)

                            for n in range(left_o):
                                stencil_entry = stencil_entry.subs(self._f[n-int(self._s_o/2)],
                                                                   l_poly.subs(self._x_c, (n-int(self._s_o/2))*self._h_x))
                        else:
                            l_poly = None

                    elif self._s_o >= 4:
                        # Available points for unified polynomial construction
                        a_p_uni = self._s_o + 1 - right_u - left_u
                        # Special case when points available for unified polynomial are zero (or smaller)
                        if a_p_uni <= 0:
                            # Grab the unified polynomial for one point
                            u_poly = self._u_poly_variants[0]
                            # Substitute u_x[0] with single available f
                            # Substitute x[0] with position*h_x
                            u_poly = u_poly.subs([(self._u_x[0], self._f[0]),
                                                  (self._x[0], 0)])

                            # If r is even, then set eta_r to 0.5*h_x
                            if ri % 2 == 0:
                                u_poly = u_poly.subs(self._x_r, 0.5*self._h_x)
                            else:
                                u_poly = u_poly.subs(self._x_r, self._eta_r*self._h_x)
                            # If l is even, then set eta_l to -0.5+*h_x
                            if le % 2 == 0:
                                u_poly = u_poly.subs(self._x_l, -0.5*self._h_x)
                            else:
                                u_poly = u_poly.subs(self._x_l, self._eta_l*self._h_x)

                            for n in range(right_o):
                                stencil_entry = stencil_entry.subs(self._f[int(self._s_o/2)-n],
                                                                   u_poly.subs(self._x_c, (int(self._s_o/2)-n)*self._h_x))
                            for n in range(left_o):
                                stencil_entry = stencil_entry.subs(self._f[n-int(self._s_o/2)],
                                                                   u_poly.subs(self._x_c, (n-int(self._s_o/2))*self._h_x))

                        else:
                            # Grab the polynomial for that number of points
                            u_poly = self._u_poly_variants[a_p_uni - 1]
                            for n in range(1+self._s_o-right_u - left_u):
                                u_poly = u_poly.subs([(self._u_x[n], self._f[left_u+n-int(self._s_o/2)]),
                                                      (self._x[n], (left_u+n-int(self._s_o/2))*self._h_x)])
                            u_poly = u_poly.subs(self._x_r, self._eta_r*self._h_x)
                            u_poly = u_poly.subs(self._x_l, self._eta_l*self._h_x)
                            for n in range(right_o):
                                stencil_entry = stencil_entry.subs(self._f[int(self._s_o/2)-n],
                                                                   u_poly.subs(self._x_c, (int(self._s_o/2)-n)*self._h_x))
                            for n in range(left_o):
                                stencil_entry = stencil_entry.subs(self._f[n-int(self._s_o/2)],
                                                                   u_poly.subs(self._x_c, (n-int(self._s_o/2))*self._h_x))
                    else:
                        # Order 2 edge case (use separate polynomials)
                        # For order 2, the double sided polynomial is never needed.
                        # Right side polynomial
                        r_poly = self._i_poly_variants

                        # Substitute in correct values of x and u_x
                        r_poly = r_poly.subs([(self._u_x[0], self._f[0]),
                                              (self._x[0], 0)])

                        # If ri is 2, then set eta_r to 0.5*h_x
                        if ri == 2:
                            r_poly = r_poly.subs(self._x_b, 0.5*self._h_x)
                        else:
                            r_poly = r_poly.subs(self._x_b, self._eta_r*self._h_x)

                        for n in range(right_o):
                            stencil_entry = stencil_entry.subs(self._f[int(self._s_o/2)-n],
                                                               r_poly.subs(self._x_c, (int(self._s_o/2)-n)*self._h_x))

                        # Left side polynomial
                        l_poly = self._i_poly_variants

                        # Substitute in correct values of x and u_x
                        l_poly = l_poly.subs([(self._u_x[0], self._f[0]),
                                              (self._x[0], 0)])

                        # If le is 2, then set eta_r to 0.5*h_x
                        if le == 2:
                            l_poly = l_poly.subs(self._x_b, -0.5*self._h_x)
                        else:
                            l_poly = l_poly.subs(self._x_b, self._eta_l*self._h_x)

                        for n in range(left_o):
                            stencil_entry = stencil_entry.subs(self._f[n-int(self._s_o/2)],
                                                               l_poly.subs(self._x_c, (n-int(self._s_o/2))*self._h_x))
                else:
                    stencil_entry = base_stencil
                self._stencil_list[le][ri] = sp.simplify(stencil_entry)

        self._stencil_dict[str(self._bcs)+str(self._s_o)+str(deriv)+'ns'] = self._stencil_list
