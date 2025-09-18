"""Test computation of eye-pose covariance.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from sensitivity_analysis import EyePoseCovariance
import sensitivity_analysis as sensitivity_analysis
from sensitivity_analysis.tests import test_utils
from parameterized import parameterized, param, parameterized_class
import torch
import unittest


@parameterized_class(
    ("with_limbus", "subsystem_idx"),
    [
        (with_limbus, subsystem_idx)
        for with_limbus in (True, False)
        for subsystem_idx in (0, 1)
    ]
)
class TestEyePoseCovariance(test_utils.TestCommonUtils):
    """Unit tests for computation of eye-pose covariance.
    """

    def setUp(self):
        """Initialize data for tests.
        """

        super().setUp()

        self.derivative_data.set_eye_and_subsystem(index=self.subsystem_idx)
        self.cov_calc = EyePoseCovariance(self.derivative_data)
        self.cov_calc.with_limbus = self.with_limbus

        # Generate data.
        self.cov_calc.prep_glint_data()
        self.num_glints = self.cov_calc.derivative_data.num_visible_glints
        self.num_eye_pose = \
            self.cov_calc.derivative_calculator_eye_pose.get_num_parameters()

        self.cov_calc.prep_pupil_data()
        self.num_pupil = \
            self.cov_calc.derivative_calculator_pupil_lifting.\
            get_num_parameters()

        if self.with_limbus:
            self.cov_calc.prep_limbus_data()
            self.num_limbus = \
                self.cov_calc.derivative_calculator_limbus_lifting.\
                get_num_parameters()
        else:
            self.num_limbus = 0

    def test_size_hessian(self):
        """Test the size of the Hessian of the cost function.
        """

        #     [d2_pose           d_pose_d_pupil                  0]
        # H = [d_pose_d_pupil.T        d2_pupil    d_pose_d_limbus]
        #     [             0.T  d_pose_d_limbus.T       d2_limbus]

        hessian = self.cov_calc.compute_hessian()

        # The hessian should always be a square matrix.
        self.assertTrue(hessian.shape[0] == hessian.shape[1])

        self.assertTrue(
            hessian.shape[0] ==
            self.num_eye_pose + self.num_pupil + self.num_limbus
        )

    def test_sparsity_hessian(self):
        """Test the sparsity of the Hessian of the cost function.
        """

        hessian = self.cov_calc.compute_hessian()

        #             a       b         c
        #     [top_left top_mid top_right] a
        # H = [mid_left mid_mid mid_right] b
        #     [bot_left bot_mid bot_right] c

        a = self.num_eye_pose
        b = a + self.num_pupil
        c = b + self.num_limbus

        top_left = hessian[:a, :a]
        self.assertFalse(torch.allclose(top_left, torch.zeros(a, a)))

        top_mid = hessian[:a, a:b]
        mid_left = hessian[a:b, :a]
        self.assertTrue(torch.allclose(top_mid, mid_left.T))

        top_right = hessian[:a, b:c]
        bot_left = hessian[b:c, :a]
        self.assertTrue(torch.allclose(top_right, bot_left.T))

        mid_right = hessian[a:b, b:c]
        bot_mid = hessian[b:c, a:b]
        self.assertTrue(torch.allclose(mid_right, bot_mid.T))
        self.assertTrue(torch.allclose(mid_right, torch.zeros(b-a, c-b)))

        bot_right = hessian[b:c, b:c]
        if self.with_limbus:
            self.assertFalse(torch.allclose(bot_right, torch.zeros(c-b, c-b)))

    @parameterized.expand(
        [param(optimize_eye_shape) for optimize_eye_shape in (True, False)]
    )
    def test_sparsity_cross_derivative(self, optimize_eye_shape):
        """Test the sparsity of the cross derivative of the cost function.

        Args:
            optimize_eye_shape (bool): whether (True) consider eye shape as
            part of the parameters that are optimized or instead (False) to
            consider it as a data input. The former is the standard setting for
            user calibration; the latter, for eye-tracking proper.
        """

        # If with_limbus and optimize_eye_shape are False, we have
        #
        # w = (w2 = eye pose, w3 = pupil lifting)
        # z = (z1 = eye shape, z2 = camera extrinsics, z3 = camera intrinsics,
        #           z4 = leds, z5 = glints, z6 = pupil).
        #
        # If with_limbus is True and optimize_eye_shape is False, we have
        #
        # w = (w2, w3, w4 = limbus lifting)
        # z = (z1, ..., z6, z7 = limbus).
        #
        # If with_limbus is False and optimize_eye_shape is True, we have
        #
        # w = (w1 = eye shape, w2, w3)
        # z = (z2, ..., z6)
        #
        # If with_limbus is True and optimize_eye_shape is True, we have
        #
        # w = (w1, ..., w4)
        # z = (z2, ...., z7)

        cross_derivative = \
            self.cov_calc.compute_cross_derivative(
                optimize_eye_shape=optimize_eye_shape
            )

        num_eye_shape = \
            self.cov_calc.derivative_calculator_eye_shape.get_num_parameters()
        num_eye_shape_cols = 0 if optimize_eye_shape else num_eye_shape

        # Pupil and limbus lifting parameters are one dimensional.
        num_rows = self.num_eye_pose + self.num_pupil + self.num_limbus

        self.assertTrue(cross_derivative.shape[0] == num_rows)

        num_extrinsics = \
            self.cov_calc.derivative_calculator_camera_extrinsics.\
            get_num_parameters()
        num_intrinsics = \
            self.cov_calc.derivative_calculator_camera_intrinsics.\
            get_num_parameters()
        num_led_params = \
            self.cov_calc.derivative_calculator_leds.get_num_parameters()

        # Glints, pupil, and limbus points are two dimensional.
        num_cols = \
            num_eye_shape_cols + num_extrinsics + num_intrinsics + \
            num_led_params + \
            2*(self.num_glints + self.num_pupil + self.num_limbus)

        self.assertTrue(cross_derivative.shape[1] == num_cols)

    def test_eye_pose_derivative(self):
        """Test computation of derivative of eye pose w.r.t. data parameters.
        """

        d_optim_d_data = self.cov_calc.compute_d_optim_d_data()

        num_eye_shape = \
            self.cov_calc.derivative_calculator_eye_shape.get_num_parameters()
        num_extrinsics = \
            self.cov_calc.derivative_calculator_camera_extrinsics.\
            get_num_parameters()
        num_intrinsics = \
            self.cov_calc.derivative_calculator_camera_intrinsics.\
            get_num_parameters()
        num_led_params = \
            self.cov_calc.derivative_calculator_leds.get_num_parameters()

        num_optim = self.num_eye_pose + self.num_pupil + self.num_limbus

        num_data = \
            num_eye_shape + \
            num_extrinsics + num_intrinsics + \
            num_led_params + \
            2 * (self.num_glints + self.num_pupil +
                 self.num_limbus)  # Two dimensional!

        self.assertTrue(d_optim_d_data.shape[0] == num_optim)
        self.assertTrue(d_optim_d_data.shape[1] == num_data)

    def test_compute_covariance(self):
        """Test computation of covariance of noise in eye-shape parameters.
        """

        cov_eye_pose = self.cov_calc.compute_covariance()

        num_pose = \
            self.cov_calc.derivative_calculator_eye_pose.get_num_parameters()
        self.assertTrue(cov_eye_pose.shape[0] == num_pose)

        # Is valid covariance.
        self.assertTrue(sensitivity_analysis.is_valid_covariance(cov_eye_pose))

    def test_compute_covariance_pupil_center(self):
        """Test computation of covariance of noise in pupil center.
        """

        self.cov_calc.compute_covariance()
        cov_pupil_inUser = self.cov_calc.compute_covariance_pupil_center()

        self.assertTrue(cov_pupil_inUser.shape[0] == 3)
        self.assertTrue(cov_pupil_inUser.shape[1] == 3)
        # Is symmetric.
        self.assertTrue(torch.allclose(cov_pupil_inUser, cov_pupil_inUser.T))

    def test_compute_error_probability_for_pupil_center(self):
        """Test computation of error probabilities of pupil center.
        """

        # Generate data.
        self.cov_calc.compute_covariance()

        # Compute error in user's coordinate system.
        max_error_mm = 3.0
        for coord in ["x", "y", "z"]:
            p_error = \
                self.cov_calc._compute_error_probability_for_pupil_center(
                    coord, max_error_mm
                )

            # p_error is decreasing, all values are no greater than 1.0.
            p_previous = 1.1
            for p_next in p_error:
                self.assertTrue(p_next <= 1.0)
                # Not strictly decreasing as it could saturate at 0 or 1.
                self.assertTrue(p_next <= p_previous)
                p_previous = p_next


if __name__ == "__main__":
    unittest.main()
