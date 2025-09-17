"""Unit tests for methods and classes in core.py.

Unit tests for methods and classes in core.py
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import torch
import unittest
from parameterized import parameterized


class TestNumeric(unittest.TestCase):
    def test_deg_to_rad(self):
        self.assertTrue(core.deg_to_rad(torch.tensor(30.0)) == core.TPI / 6)
        self.assertTrue(core.deg_to_rad(torch.tensor(60.0)) == core.TPI / 3)
        self.assertTrue(core.deg_to_rad(torch.tensor(90.0)) == core.TPI / 2)
        self.assertTrue(core.deg_to_rad(torch.tensor(-30.0)) == -core.TPI / 6)
        self.assertTrue(core.deg_to_rad(torch.tensor(-60.0)) == -core.TPI / 3)
        self.assertTrue(core.deg_to_rad(torch.tensor(-90.0)) == -core.TPI / 2)

    def test_rad_to_deg(self):
        self.assertTrue(core.rad_to_deg(core.TPI / 6) == torch.tensor(30.0))
        self.assertTrue(core.rad_to_deg(core.TPI / 3) == torch.tensor(60.0))
        self.assertTrue(core.rad_to_deg(core.TPI / 2) == torch.tensor(90.0))
        self.assertTrue(core.rad_to_deg(-core.TPI / 6) == torch.tensor(-30.0))
        self.assertTrue(core.rad_to_deg(-core.TPI / 3) == torch.tensor(-60.0))
        self.assertTrue(core.rad_to_deg(-core.TPI / 2) == torch.tensor(-90.0))

    def test_stack_tensors(self):
        for N in range(3, 5):
            zero_hom = torch.hstack((torch.zeros(N), core.T1.clone()))
            tuple_of_rows = [zero_hom for _ in range(N + 1)]
            zeros = core.stack_tensors(tuple_of_rows)
            shape = zeros.shape
            self.assertTrue(shape[0] == N + 1)
            self.assertTrue(shape[1] == N + 1)
            self.assertTrue(zeros[:, :N].max() == core.T0.clone())
            self.assertTrue(zeros[:, :N].min() == core.T0.clone())
            self.assertTrue(zeros[:, N].max() == core.T1.clone())
            self.assertTrue(zeros[:, N].min() == core.T1.clone())

    def test_compute_numeric_jacobian_from_tensors(self):
        """test_compute_numeric_jacobian_from_tensors.

        Test computation of Jacobians numerically and using Autograd.
        """
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        def fun(x):
            return torch.outer(x, x)

        y = fun(x)
        dy_dx_numeric = core.compute_numeric_jacobian_from_tensors(x, fun)
        dy_dx_autograd = core.compute_auto_jacobian_from_tensors(y, x)

        self.assertTrue(torch.allclose(dy_dx_autograd, dy_dx_numeric))

    @parameterized.expand([
        ("cpu",),
        ("cuda",) if torch.cuda.is_available() else ("cpu",),
    ])
    def test_compute_numeric_jacobian_device_parameter(self, device):
        """Test compute_numeric_jacobian_from_tensors with device parameter."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        def test_func(x):
            return torch.sum(x ** 2)
        
        result = core.compute_numeric_jacobian_from_tensors(x, test_func, device=device)
        
        self.assertEqual(result.device.type, device)
        
        expected = 2 * x
        if device != x.device.type:
            expected = expected.to(device)
        self.assertTrue(torch.allclose(result.squeeze(), expected, rtol=1e-4))

    @parameterized.expand([
        ("cpu",),
        ("cuda",) if torch.cuda.is_available() else ("cpu",),
    ])
    def test_compute_auto_jacobian_device_parameter(self, device):
        """Test compute_auto_jacobian_from_tensors with device parameter."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.sum(x ** 2)
        
        result = core.compute_auto_jacobian_from_tensors(y, x, device=device)
        
        self.assertEqual(result.device.type, device)
        
        expected = 2 * x
        if device != x.device.type:
            expected = expected.to(device)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-4))

    @parameterized.expand([
        ("cpu",),
        ("cuda",) if torch.cuda.is_available() else ("cpu",),
    ])
    def test_alt_compute_auto_jacobian_device_parameter(self, device):
        """Test alt_compute_auto_jacobian_from_tensors with device parameter."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.sum(x ** 2)
        
        dy_dx = core.compute_auto_jacobian_from_tensors(y, x, create_graph=True)
        
        result = core.alt_compute_auto_jacobian_from_tensors(dy_dx, x, device=device)
        
        self.assertEqual(result.device.type, device)
        
        expected = 2 * torch.eye(len(x))
        if device != x.device.type:
            expected = expected.to(device)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-4))

    def test_device_parameter_consistency(self):
        """Test that all three functions produce consistent results with device parameter."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for consistency test")
        
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        def test_func(x):
            return torch.sum(x ** 2)
        
        cpu_numeric = core.compute_numeric_jacobian_from_tensors(x, test_func, device="cpu")
        cuda_numeric = core.compute_numeric_jacobian_from_tensors(x, test_func, device="cuda")
        
        y = test_func(x)
        cpu_auto = core.compute_auto_jacobian_from_tensors(y, x, device="cpu")
        cuda_auto = core.compute_auto_jacobian_from_tensors(y, x, device="cuda")
        
        # Verify results are consistent across devices
        self.assertTrue(torch.allclose(cpu_numeric.cpu(), cuda_numeric.cpu(), rtol=1e-4))
        self.assertTrue(torch.allclose(cpu_auto.cpu(), cuda_auto.cpu(), rtol=1e-4))
        self.assertTrue(torch.allclose(cpu_auto.cpu(), cpu_numeric.squeeze().cpu(), rtol=1e-4))

    def test_compute_auto_jacobian_from_tensors(self):
        # Test computation of first derivatives:
        N = 3
        x = torch.ones(N, requires_grad=True)
        A = core.stack_tensors(
            [torch.tensor(range(i, i + N), dtype=x.dtype) for i in range(N)]
        )
        y = (1 / 2) * (x @ A @ x) + 1
        # create_graph is set to true so that we can compute derivative of
        # output, i.e., second derivatives.
        dy_dx = core.compute_auto_jacobian_from_tensors(
            y, x, create_graph=True)
        self.assertTrue(torch.allclose(dy_dx, A @ x))

        # Test computation of second derivatives
        d2y_dx2 = core.compute_auto_jacobian_from_tensors(dy_dx, x)
        self.assertTrue(torch.allclose(d2y_dx2, A))

    def test_poly_solver(self):
        # Create 1st degree polynomial.
        p = torch.tensor([-2.0, 2], requires_grad=True)  # -2 + 2*x
        roots = core.poly_solver(p)
        roots_expected = -(p[0] / p[1]).reshape(1)

        self.assertTrue(
            torch.allclose(
                roots,
                roots_expected,
                rtol=core.EPS,
                atol=core.EPS
            )
        )

        droots_dp = core.compute_auto_jacobian_from_tensors(
            roots, p, create_graph=True
        )
        # Derivative of root with respect to coefficients is [-1/p[1],
        # p[0]/p[1]^2]
        droots_dp_expected = (torch.stack(
            (-core.T1, p[0] / p[1])) / p[1]).reshape(1, 2)

        self.assertTrue(
            torch.allclose(
                droots_dp,
                droots_dp_expected,
                rtol=core.EPS,
                atol=core.EPS
            )
        )

        # Need to fall back on old, inefficient way, because some methods have
        # not yet been designed to work with batch computations in
        # torch.autograd.grad.
        d2roots_dp2 = core.alt_compute_auto_jacobian_from_tensors(
            droots_dp, p
        )
        # Second derivative of root with respect to coefficients is [[0,
        # 1/p[1]^2], [1/p[1]^2, -2*p[0]/p[1]^3]]
        d2roots_dp2_expected = core.stack_tensors(
            (
                torch.stack((core.T0, 1 / p[1]**2)),
                torch.stack((1 / p[1]**2, -2 * p[0] / p[1]**3))
            )
        )

        self.assertTrue(
            torch.allclose(
                d2roots_dp2.view(2, 2),
                d2roots_dp2_expected,
                rtol=core.EPS,
                atol=core.EPS
            )
        )

        # Create 2nd degree polynomial.
        p = torch.tensor([1.0, -2.0, 1.0])  # 1 - 2*x + x^2
        roots = core.poly_solver(p)
        roots_expected = torch.tensor((1.0, 1.0))

        self.assertTrue(
            torch.allclose(
                roots,
                roots_expected,
                rtol=core.EPS,
                atol=core.EPS
            )
        )

        # Create 3rd degree polynomial. (-1 + x) * (-2 + x) * (-3 + x) * 2 =
        # -12 + 22x - 12x^2 + 2x^3
        p = torch.tensor([-12.0, 22.0, -12.0, 2.0])
        roots = torch.sort(core.poly_solver(p))[0]
        roots_expected = torch.tensor((1.0, 2.0, 3.0))

        self.assertTrue(
            torch.allclose(
                roots,
                roots_expected,
                rtol=core.EPS * 100,  # Needs some leeway.
                atol=core.EPS * 100
            )
        )

        # Create 4th degree polynomial. (-1 + x) * (-2 + x) * (-3 + x) * (-4 +
        # x) * 2 = 48 - 100x + 70x^2 - 20x^3 + 2x^4
        p = torch.tensor([48.0, -100, 70, -20, 2], requires_grad=True)
        roots = torch.sort(core.poly_solver(p))[0]
        roots_expected = torch.tensor((1.0, 2, 3, 4))

        self.assertTrue(
            torch.allclose(
                roots,
                roots_expected,
                rtol=core.EPS * 100,  # Needs some leeway.
                atol=core.EPS * 100
            )
        )

        # Create simple 4th degree polynomial so that we can easily compute the
        # derivative of its roots with respect to its coefficients.
        #
        # x*(x - c1)*(x - c2)*(x - c3) = 0 - c1*c2*c3*x + (c1*c2 + c1*c3 +
        #   c2*c3)*x^2 - (c1 + c2 + c3)*x^3 + x^4
        #
        # The roots are 0, c1, c2, and c3
        c1 = torch.tensor(1.0, requires_grad=True)
        c2 = torch.tensor(2.0, requires_grad=True)
        c3 = torch.tensor(3.0, requires_grad=True)

        # This yields p = 0 - 6x + 11x^2 - 6x^3 + x4
        p = torch.stack(
            (
                core.T0,
                -c1 * c2 * c3,
                c1 * c2 + c1 * c3 + c2 * c3,
                -c1 - c2 - c3,
                core.T1
            )
        )
        roots = torch.sort(core.poly_solver(p))[0]
        roots_expected = torch.stack((core.T0, c1, c2, c3))

        self.assertTrue(
            torch.allclose(
                roots,
                roots_expected,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

        droots_dp = core.compute_auto_jacobian_from_tensors(
            roots, p, create_graph=True
        )
        # The derivative of the roots (c0=0, c1=1, c2=2, c3=3) with respect to
        # some of the coefficients can be computed in closed form using sympy
        # as such:
        #
        # import sympy
        #
        # def root(a, b, c, d, k=0): eps = (-1 + sympy.sqrt(-3)) / 2 eps_k =
        #     eps**k D0 = b**2 - 3*a*c D1 = 2*(b**3) - 9*a*b*c + 27*(a**2)*d C
        #     = sympy.real_root( ( D1 + \
        #                 sympy.real_root(
        #                     D1**2 - 4*(D0**3),
        #                     2
        #                 )
        #             ) / 2,
        #             3
        #         )
        #
        #     result = -(b + eps_k*C + D0/C/eps_k)/3/a
        #     result = \
        #       (result.conjugate().conjugate() + result.conjugate()) / 2
        #     return result
        #
        # def Da(a, b, c, d): result = sympy.diff(root(t, b, c, d), t).subs(t,
        #     a) result = \
        #       (result.conjugate().conjugate() + result.conjugate()) / 2
        #     return sympy.simplify(result)
        #
        # def Db(a, b, c, d): result =  sympy.diff(root(a, t, c, d), t).subs(t,
        #     b) result = \
        #       (result.conjugate().conjugate() + result.conjugate()) / 2
        #     return sympy.simplify(result)
        #
        # def Dc(a, b, c, d): result = sympy.diff(root(a, b, t, d), t).subs(t,
        #     c) result = \
        #       (result.conjugate().conjugate() + result.conjugate()) / 2
        #     return sympy.simplify(result)
        #
        # def Dd(a, b, c, d): result = sympy.diff(root(a, b, c, t), t).subs(t,
        #     d) result = \
        #       (result.conjugate().conjugate() + result.conjugate()) / 2
        #     return sympy.simplify(result)
        #
        # t, a, b, c, d = sympy.symbols("t a b c d", real=True)
        #
        # print("root:", sympy.simplify(root(1, -6, 11, -6))) print("*" * 20)
        # print("droot_da:", Da(1, -6, 11, -6)) print("droot_db:", Db(1, -6,
        # 11, -6)) print("droot_dc:", Dc(1, -6, 11, -6)) print("droot_dd:",
        # Dd(1, -6, 11, -6))
        #
        # Using the notation p = (p0, p1, p2, p3, p4), we have
        #
        # dc0_d(p1, p2, p3, p4) = Don't know. dc1_d(p1, p2, p3, p4) = (-1/2,
        # -1/2, -1/2,  -1/2) dc2_d(p1, p2, p3, p4) = (   1,    2,    4,     8)
        # dc3_d(p1, p2, p3, p4) = (-1/2, -3/2, -9/2, -27/2)
        droots_123_dp_1234_expected = torch.tensor(
            (
                (-0.5, -1 / 2, -1 / 2, -1 / 2),
                (1, 2, 4, 8),
                (-1 / 2, -3 / 2, -9 / 2, -27 / 2)
            )
        )
        self.assertTrue(
            torch.allclose(
                droots_dp[1:, 1:],
                droots_123_dp_1234_expected,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

    def test_constants(self):
        self.assertTrue(core.T1 + core.TEPS / 3 == core.T1.clone())


if __name__ == "__main__":
    unittest.main()
