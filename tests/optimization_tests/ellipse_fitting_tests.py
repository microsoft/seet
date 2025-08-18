"""ellipse_fitting_tests.py

Tests for ellipse-fitting methods
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.optimization as optimization
import seet.primitives as primitives
import torch
import unittest


class TestEllipseFitting(unittest.TestCase):
    """TestEllipseFitting.

    Unit tests for ellipse fitting.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """

        super().setUp()

        self.plane = \
            primitives.Plane(None, core.SE3.create_identity())

        self.center_inParent2DPlane = torch.tensor([1.0, 1.0])
        self.angle_deg = torch.tensor(30.0)
        self.x_radius = torch.tensor(1.0)
        self.y_radius = torch.tensor(2.0)

        self.gt_ellipse = \
            primitives.Ellipse.create_from_origin_angle_and_axes_inPlane(
                self.plane,
                self.center_inParent2DPlane,
                self.angle_deg,
                self.x_radius,
                self.y_radius
            )

        # Sampled points on ellipse.
        points_in2DPlane, _ = self.gt_ellipse.get_points_inParent()
        self.points_in2DPlane = points_in2DPlane[:2, :]

        self.ellipse_fitter = \
            optimization.EllipseFitting(self.plane, self.points_in2DPlane.T)

    def compare_ellipses(
        self, ellipse, rtol=core.TEPS.item() * 100, atol=core.TEPS.item() * 100
    ):
        """
        Perform comparison between given ellipse and ground truth.

        Args:
            ellipse (Ellipse): ellipse to compare against GT
        """

        # Center is good.
        center_inParent3DPlane, _ = \
            ellipse.get_center_and_normal_inParent()
        self.assertTrue(
            torch.allclose(
                center_inParent3DPlane[:2],
                self.center_inParent2DPlane,
                rtol=rtol,
                atol=atol
            )
        )
        # Axes are good. Watch out for that flip!
        dist_xx = torch.linalg.norm(ellipse.x_radius - self.x_radius)
        dist_xy = torch.linalg.norm(ellipse.x_radius - self.y_radius)
        if dist_xy < dist_xx:
            estimate_x_radius = ellipse.y_radius
            estimate_y_radius = ellipse.x_radius
        else:
            estimate_x_radius = ellipse.x_radius
            estimate_y_radius = ellipse.y_radius

        self.assertTrue(
            torch.allclose(
                estimate_x_radius, self.x_radius, rtol=rtol, atol=atol
            )
        )
        self.assertTrue(
            torch.allclose(
                estimate_y_radius, self.y_radius, rtol=rtol, atol=atol
            )
        )

    def test_cost(self):
        """test_cost.

        Test whether the cost function is working.
        """

        # Cost at ground truth should be nearly zero.
        self.ellipse_fitter.set_initial_values(
            self.center_inParent2DPlane,
            self.angle_deg,
            self.x_radius,
            self.y_radius
        )
        parameter = self.ellipse_fitter._torch_to_array()
        cost = self.ellipse_fitter.cost_function(parameter)
        self.assertAlmostEqual(float(cost), 0.0)

        # Derivative of cost at ground truth should be nearly zero.
        d_cost = self.ellipse_fitter.jacobian_cost_function(parameter)
        self.assertTrue(
            torch.allclose(
                torch.tensor(d_cost),
                torch.zeros(d_cost.shape),
                rtol=core.TEPS.item() * 100,
                atol=core.TEPS.item() * 100
            )
        )

        # Cost away from ground truth should be large.
        parameter = 10 * parameter
        self.ellipse_fitter._array_to_torch(parameter)
        self.ellipse_fitter.set_initial_values(
            self.center_inParent2DPlane,
            self.angle_deg,
            self.x_radius,
            self.y_radius
        )
        cost = self.ellipse_fitter.cost_function(parameter)
        self.assertGreater(float(cost), 0.0)

        # Derivative of cost at ground truth should be nearly zero.
        d_cost = self.ellipse_fitter.jacobian_cost_function(parameter)
        self.assertFalse(
            torch.allclose(
                torch.tensor(d_cost),
                torch.zeros(d_cost.shape),
                rtol=core.TEPS.item() * 100,
                atol=core.TEPS.item() * 100
            )
        )

    def test_fit(self):
        """test_fit.

        Test fitting of ellipse.
        """

        # Fitting of ellipse initialized at ground truth should return gt.
        self.ellipse_fitter.set_initial_values(
            self.center_inParent2DPlane,
            self.angle_deg,
            self.x_radius,
            self.y_radius
        )
        ellipse_estimate = self.ellipse_fitter.fit(options={"disp": True})
        self.compare_ellipses(ellipse_estimate)

        # Fitting an ellipse initialized away from gt should still return gt.
        # However, there is ambiguity in the parameterization of an ellipse:
        # swapping the length of the semi-axes and rotating the ellipse by 90
        # deg produces the same ellipse. We need to watch out for that!

        self.ellipse_fitter.initialize(method="svd")
        ellipse_estimate = self.ellipse_fitter.fit(options={"disp": True})
        self.compare_ellipses(ellipse_estimate)

        self.ellipse_fitter.initialize(method="algebraic")
        ellipse_estimate = self.ellipse_fitter.fit(options={"disp": True})
        self.compare_ellipses(ellipse_estimate)

    def test_fit_hard(self):
        """test_fit_hard.

        Harder tests for ellipse fitting.
        """

        # Omit half of the points.
        num_points = self.points_in2DPlane.shape[1]
        fewer_points_in2DPlane = self.points_in2DPlane[:, :num_points // 2]

        self.ellipse_fitter = \
            optimization.EllipseFitting(self.plane, fewer_points_in2DPlane.T)

        self.ellipse_fitter.initialize(method="svd")
        ellipse_estimate = self.ellipse_fitter.fit(options={"disp": True})
        self.compare_ellipses(ellipse_estimate)

        # Omit 75% of the points.
        # Omit half of the points.
        num_points = fewer_points_in2DPlane.shape[1]
        fewer_points_in2DPlane = fewer_points_in2DPlane[:, :num_points // 2]

        assert \
            (fewer_points_in2DPlane.shape[1] > 4), \
            "Cannot fit ellipse to fewer than 5 points"

        self.ellipse_fitter = \
            optimization.EllipseFitting(self.plane, fewer_points_in2DPlane.T)

        self.ellipse_fitter.initialize(method="algebraic")
        ellipse_estimate = self.ellipse_fitter.fit(options={"disp": True})
        atol = rtol = 1e-2  # Numerics are really tough with only a few points.
        self.compare_ellipses(ellipse_estimate, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
