import pytest

import numpy as np
import sympy as sp
import pickle
import os

from devitoboundary.stencils.stencils import (taylor, BoundaryConditions, get_ext_coeffs,
                                              get_stencils, get_stencils_lambda)
from devitoboundary.symbolics.symbols import x_a, x_t, x_b, E


class TestBCs:
    """Tests for the BoundaryConditions object"""
    @pytest.mark.parametrize('order, expected',
                             [(2, '(x - x_b)**2*a[2] + (x - x_b)*a[1] + a[0]'),
                              (4, '(x - x_b)**4*a[4] + (x - x_b)**3*a[3]'
                                  ' + (x - x_b)**2*a[2] + (x - x_b)*a[1] + a[0]'),
                              (6, '(x - x_b)**6*a[6] + (x - x_b)**5*a[5]'
                                  ' + (x - x_b)**4*a[4] + (x - x_b)**3*a[3]'
                                  ' + (x - x_b)**2*a[2] + (x - x_b)*a[1] + a[0]'),
                              (8, '(x - x_b)**8*a[8] + (x - x_b)**7*a[7]'
                                  ' + (x - x_b)**6*a[6] + (x - x_b)**5*a[5]'
                                  ' + (x - x_b)**4*a[4] + (x - x_b)**3*a[3]'
                                  ' + (x - x_b)**2*a[2] + (x - x_b)*a[1] + a[0]')])
    def test_taylor(self, order, expected):
        """Test generated Taylor series are as expected"""
        x = sp.symbols('x')
        series = taylor(x, order)
        assert str(series) == expected

    @pytest.mark.parametrize('spec, order, expected',
                             [({0: 0, 2: 0, 4: 0}, 4, '(x - x_b)**3*a[3] + (x - x_b)*a[1]'),
                              ({1: 0, 3: 0, 5: 0}, 2, '(x - x_b)**2*a[2] + a[0]'),
                              ({0: 0, 2: 0, 4: 0, 6: 0}, 6, '(x - x_b)**5*a[5] + (x - x_b)**3*a[3]'
                                                            ' + (x - x_b)*a[1]'),
                              ({1: 0, 3: 0, 5: 0}, 4, '(x - x_b)**4*a[4] + (x - x_b)**2*a[2] + a[0]'),
                              ({1: 10}, 1, '10.0*x - 10.0*x_b + a[0]')])
    def test_get_taylor(self, spec, order, expected):
        """Test Taylor series accounting for bcs are as expected"""
        bcs = BoundaryConditions(spec, order)

        series = bcs.get_taylor(order=None)

        assert str(series) == expected


class TestExtrapolations:
    """Tests for the generated extrapolations"""
    @pytest.mark.parametrize('spec, order, coeff, expected',
                             [({0: 0, 2: 0, 4: 0}, 4, 0, '(zeta-j)*j*(j-2*zeta)/((1+2*zeta)*(1+zeta))'),
                              ({0: 0, 2: 0, 4: 0}, 4, 1, '(zeta-j)*(1+j)*(1+2*zeta-j)/(zeta*(1+2*zeta))'),
                              ({1: 0, 3: 0}, 4, 0, 'j*(j-2*zeta)/(1+2*zeta)'),
                              ({1: 0, 3: 0}, 4, 1, '(1+j)*(1+2*zeta-j)/(1+2*zeta)'),
                              ({0: 0, 2: 0, 4: 0, 6: 0}, 6, 1, 'j*(j-zeta)*(j+2)*(j-2*zeta)*(j-2*zeta-2)/((1+zeta)*(2*zeta + 1)*(2*zeta+3))'),
                              ({1: 0, 3: 0, 5: 0}, 6, 1, '-j*(j+2)*(j-2*zeta-2)*(j-2*zeta)/((2*zeta+1)*(2*zeta+3))')])
    def test_get_ext_coeffs(self, spec, order, coeff, expected):
        """
        Test to check that extrapolation coefficnts are generated as expected, and
        match known results.
        """
        zeta, j = sp.symbols('zeta, j')

        # Substitutions needed
        test_subs = [(x_a[i], 1-order//2+i) for i in range(order//2)]
        test_subs += [(x_t, j), (x_b, zeta)]

        bcs = BoundaryConditions(spec, order)

        coeffs = get_ext_coeffs(bcs)[order//2]

        expected_coeff = sp.sympify(expected, locals={'zeta': zeta, 'j': j})

        generated_coeff = coeffs[E[coeff]].subs(test_subs)

        assert sp.simplify(generated_coeff - expected_coeff) == 0

    def test_coefficient_orders(self):
        """
        Test to check that the lower-order coefficients generated are consistent
        """
        for i in range(2, 5):
            spec = {2*j: 0 for j in range(i)}
            bcs_ref = BoundaryConditions(spec, 2*i-2)
            bcs_main = BoundaryConditions(spec, 2*i)

            coeffs_ref = get_ext_coeffs(bcs_ref)[i-1]
            coeffs_main = get_ext_coeffs(bcs_main)[i-1]

            assert coeffs_ref == coeffs_main

    @pytest.mark.parametrize('order', [4, 6])
    @pytest.mark.parametrize('type', ['even', 'odd'])
    def test_polynomial_recovery(self, order, type):
        """
        Test that polynomials of matching order are correctly recovered
        """
        if type == 'even':
            spec = {2*i: 0 for i in range(order)}
        else:
            spec = {2*i+1: 0 for i in range(order)}
        bcs = BoundaryConditions(spec, order)

        coeffs = get_ext_coeffs(bcs)[order//2]

        poly = bcs.get_taylor(order=order-1)

        extrapolation = sum([coeffs[E[i]]*poly.subs(bcs.x, x_a[i]) for i in range(order//2)])
        exterior = poly.subs(bcs.x, x_t)

        assert sp.simplify(extrapolation - exterior) == 0

    @pytest.mark.parametrize('order', [2, 4])
    def test_caching_write(self, order):
        """Test that caching writes correctly"""
        spec = {2*i: 0 for i in range(order)}
        bcs = BoundaryConditions(spec, order)

        # Reset the test cache
        with open('tests/test_extrapolation_cache_w.dat', 'wb') as f:
            pickle.dump({}, f)

        # Write an extrapolation
        write_extrapolation = get_ext_coeffs(bcs, cache='tests/test_extrapolation_cache_w.dat')

        # Write an extrapolation of order+2
        high_spec = {2*i: 0 for i in range(order+2)}
        high_bcs = BoundaryConditions(high_spec, order+2)

        high_write_extrapolation = get_ext_coeffs(high_bcs, cache='tests/test_extrapolation_cache_w.dat')

        # Read both extrapolations again and check
        cached_extrapolation = get_ext_coeffs(bcs, cache='tests/test_extrapolation_cache_w.dat')
        high_cached_extrapolation = get_ext_coeffs(high_bcs, cache='tests/test_extrapolation_cache_w.dat')

        assert write_extrapolation == cached_extrapolation
        assert high_write_extrapolation == high_cached_extrapolation

    @pytest.mark.parametrize('order', [2, 4])
    def test_caching_read(self, order):
        """Test that caching reads correctly"""

        spec = {2*i: 0 for i in range(order)}
        bcs = BoundaryConditions(spec, order)

        cached_extrapolation = get_ext_coeffs(bcs, cache='tests/test_extrapolation_cache_r.dat')
        generated_extrapolation = get_ext_coeffs(bcs)
        for npts in cached_extrapolation:
            for key in cached_extrapolation[npts]:
                diff = cached_extrapolation[npts][key] - generated_extrapolation[npts][key]
                assert sp.simplify(diff) == 0


class TestStencils:
    """Tests for the modified stencils"""
    @pytest.mark.parametrize('order', [4, 6, 8])
    @pytest.mark.parametrize('derivative', [1, 2])
    def test_single_sided(self, order, derivative):
        """
        Test to check that single-sided stencils adequately approximate the
        original derivative.
        """
        # Accuracy
        thres = 0.002
        # Note: dx = 1 for simplicity

        def u_func(x, eta, deriv=0):
            if deriv == 0:
                return np.sin((x + 3*eta)*np.pi/(4*eta))
            elif deriv == 1:
                return np.pi/(4*eta)*np.cos((x + 3*eta)*np.pi/(4*eta))
            elif deriv == 2:
                return -(np.pi/(4*eta))**2*np.sin((x + 3*eta)*np.pi/(4*eta))

        spec = {2*i: 0 for i in range(order)}
        bcs = BoundaryConditions(spec, order)

        stencils_lambda = get_stencils_lambda(derivative, 0, bcs)

        errors = []

        # As left and right single sided are mirrored, fix left variant at 0
        # Skip last variant, as it is usually not too accurate
        for var in range(1, order):
            # Set max and min etas for the variant
            # Will have 9 (10+1-2) etas per variant
            min_eta = order//2 - 0.5*var + 0.05
            max_eta = order//2 - 0.5*(var-1) - 0.05
            eta = np.linspace(min_eta, max_eta, 9)[::-1]

            stencil = stencils_lambda[0, var]

            for eta_val in eta:
                evaluated = 0
                for coeff in range(order+1):
                    func = stencil[coeff]
                    multiplier = u_func(coeff-order//2, eta_val)
                    evaluated += multiplier*func(0, eta_val)
                err = abs(evaluated-u_func(0, eta_val, deriv=derivative))
                errors.append(err)

        assert np.median(errors) < thres

    @pytest.mark.parametrize('offset', [0.5, -0.5])
    @pytest.mark.parametrize('bc_type', ['even', 'odd'])
    @pytest.mark.parametrize('order', [4, 6])
    def test_special_variants(self, offset, bc_type, order):
        """
        Check that special stencil variants generated for cases where the staggered
        and unstaggered points lie on either side of the boundary are generated.
        """
        # TODO: Possibly make this test more robust
        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        # Staggered stencils only really used with first derivatives
        deriv = 1

        if bc_type == 'even':
            bcs = BoundaryConditions({2*i: 0 for i in range(1+order//2)}, order)
        else:
            bcs = BoundaryConditions({2*i + 1: 0 for i in range(1+order//2)}, order)

        stencils = get_stencils(deriv, offset, bcs, cache=cache)

        if offset == -0.5:
            assert stencils.shape == (order+1, order+2, order+1)
        elif offset == 0.5:
            assert stencils.shape == (order+2, order+1, order+1)

        # raise NotImplementedError
