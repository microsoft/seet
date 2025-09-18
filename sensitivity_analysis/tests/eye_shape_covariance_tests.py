"""Test computation of eye-shape covariance.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import sensitivity_analysis as sensitivity_analysis
from sensitivity_analysis import EyeShapeCovariance
from sensitivity_analysis.tests import \
    test_utils, \
    sensitivity_analysis_tests_configs
import os
from parameterized import parameterized_class
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
class TestEyeShapeCovariance(test_utils.TestCommonUtils):
    """Unit tests for computation of eye-shape covariance.
    """

    def setUp(self):
        """Initialize data for tests.
        """

        super().setUp()

        self.derivative_data.set_eye_and_subsystem(index=self.subsystem_idx)
        self.cov_calc = EyeShapeCovariance(self.derivative_data)
        self.cov_calc.with_limbus = self.with_limbus

        # Generate data.
        self.cov_calc.prep_glint_data()
        self.num_glints = self.cov_calc.derivative_data.num_visible_glints
        self.num_eye_pose = \
            self.cov_calc.derivative_calculator_eye_pose.get_num_parameters()
        self.num_eye_shape = \
            self.cov_calc.derivative_calculator_eye_shape.get_num_parameters()

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

        self.cov_calc._partial_compute_hessian()

        #                N1                                 N2 <- columns
        # H = [H11 (shape)^2         | H12 (shape) x (pose, lifting)] N1
        #     [H21 (pose, lifting) x (shape) | H22 (pose, lifting)^2] N2

        N1 = self.num_eye_shape

        self.assertTrue(self.cov_calc.H1_11.shape[0] == N1)  # type: ignore
        self.assertTrue(self.cov_calc.H1_11.shape[1] == N1)  # type: ignore

        N2 = self.num_eye_pose + self.num_pupil + self.num_limbus

        self.assertTrue(self.cov_calc.H1_21.shape[0] == N2)  # type: ignore
        self.assertTrue(self.cov_calc.H1_21.shape[1] == N1)  # type: ignore
        self.assertTrue(self.cov_calc.H1_22.shape[0] == N2)  # type: ignore
        self.assertTrue(self.cov_calc.H1_22.shape[1] == N2)  # type: ignore

        # If we compute the hessian again, it assumes that new data is
        # available, and the shape grows.
        self.cov_calc._partial_compute_hessian()

        # Top-left block does not change.
        self.assertTrue(self.cov_calc.H1_11.shape[0] == N1)  # type: ignore
        self.assertTrue(self.cov_calc.H1_11.shape[1] == N1)  # type: ignore

        # Other blocks increase by N2.
        self.assertTrue(
            self.cov_calc.H1_21.shape[0] == N2 + N2
        )  # type: ignore
        self.assertTrue(
            self.cov_calc.H1_21.shape[1] == N1
        )  # type: ignore
        self.assertTrue(
            self.cov_calc.H1_22.shape[0] == N2 + N2
        )  # type: ignore
        self.assertTrue(
            self.cov_calc.H1_22.shape[1] == N2 + N2
        )  # type: ignore

    def test_sparsity_hessian(self):
        """Test the sparsity of the Hessian of the cost function.
        """

        self.cov_calc._partial_compute_hessian()

        # H1_11 -> shape x shape
        # H1_21 -> (pose, pupil lift., limbus lift.) x shape
        # H1_12 = H1_21.T
        # H1_22 -> (pose, pupil lift., limbus lift.) x
        #                           (pose, pupil lift., limbus lift.)

        # H1_11 is not zero.
        self.assertFalse(
            torch.allclose(
                self.cov_calc.H1_11,  # type: ignore
                torch.zeros(self.num_eye_shape, self.num_eye_shape)
            )
        )

        #             a       b         c
        #         [top_left top_mid top_right] a
        # H1_22 = [mid_left mid_mid mid_right] b
        #         [bot_left bot_mid bot_right] c
        a = self.num_eye_pose
        b = a + self.num_pupil
        c = b + self.num_limbus

        # H1_21 is not zero.
        self.assertFalse(
            torch.allclose(
                self.cov_calc.H1_21,  # type: ignore
                torch.zeros(c, self.num_eye_shape)
            )
        )

        top_left = self.cov_calc.H1_22[:a, :a]
        self.assertFalse(torch.allclose(top_left, torch.zeros(a, a)))

        top_mid = self.cov_calc.H1_22[:a, a:b]
        mid_left = self.cov_calc.H1_22[a:b, :a]
        self.assertTrue(torch.allclose(top_mid, mid_left.T))

        top_right = self.cov_calc.H1_22[:a, b:c]
        bot_left = self.cov_calc.H1_22[b:c, :a]
        self.assertTrue(torch.allclose(top_right, bot_left.T))

        mid_right = self.cov_calc.H1_22[a:b, b:c]
        bot_mid = self.cov_calc.H1_22[b:c, a:b]
        self.assertTrue(torch.allclose(mid_right, bot_mid.T))
        self.assertTrue(torch.allclose(mid_right, torch.zeros(b - a, c - b)))

        bot_right = self.cov_calc.H1_22[b:c, b:c]
        if self.with_limbus:
            self.assertFalse(torch.allclose(
                bot_right, torch.zeros(c - b, c - b)))

    def test_sparsity_cross_derivative(self):
        """Test the sparsity of the cross derivative of the cost function.
        """

        # If with_limbus is False we have
        #
        # w = (w1 = eye_pose, w2 = eye shape, w3 = pupil lifting)
        # z = (z2 = leds, z3 = camera extrinsics, z4 = camera intrinsics,
        #                                             z5 = glints, z6 = pupil).
        #
        # If with_limbus is True we have
        #
        # w = (w1, w2, w3, w4 = limbus lifting)
        # z = (z2, ..., z6, z7 = limbus).

        self.cov_calc._partial_compute_cross_derivative()

        # Top
        num_rows = \
            self.cov_calc.derivative_calculator_eye_shape.get_num_parameters()

        self.assertTrue(
            self.cov_calc.D1_11.shape[0] == num_rows  # type: ignore
        )
        self.assertTrue(self.cov_calc.D1_12.shape[0] == num_rows)

        num_led_params = \
            self.cov_calc.derivative_calculator_leds.get_num_parameters()
        num_extrinsics = \
            self.cov_calc.derivative_calculator_camera_extrinsics.\
            get_num_parameters()
        num_intrinsics = \
            self.cov_calc.derivative_calculator_camera_intrinsics.\
            get_num_parameters()
        num_cols = num_led_params + num_extrinsics + num_intrinsics

        self.assertTrue(
            self.cov_calc.D1_11.shape[1] == num_cols  # type: ignore
        )
        self.assertTrue(self.cov_calc.D1_1.shape[1] == num_cols)

        # Bottom
        num_rows = self.num_eye_pose + self.num_pupil + self.num_limbus

        self.assertTrue(self.cov_calc.D1_1.shape[0] == num_rows)
        self.assertTrue(self.cov_calc.D1_2.shape[0] == num_rows)

        # Glints, pupil, and limbus points are two dimensional.
        num_cols = 2 * (self.num_glints + self.num_pupil + self.num_limbus)

        self.assertTrue(self.cov_calc.D1_12.shape[1] == num_cols)
        self.assertTrue(self.cov_calc.D1_2.shape[1] == num_cols)

    def test_compute_d_optim_d_data(self):
        """Test deriv. of eye-shape, pose, and lifting params. w.r.t. data.
        """

        # Create a list of gaze directions.
        list_gaze_rotation_deg = [torch.tensor([15.0, -10.0]), ]
        self.cov_calc.set_list_of_gaze_angles(list_gaze_rotation_deg)
        d_optim_d_data = self.cov_calc.compute_d_optim_d_data()

        # Need to be recomputed after
        num_eye_shape = \
            self.cov_calc.derivative_calculator_eye_shape.get_num_parameters()
        num_eye_pose = \
            self.cov_calc.derivative_calculator_eye_pose.get_num_parameters()
        num_pupil_lifting = \
            self.cov_calc.derivative_calculator_pupil_lifting.\
            get_num_parameters()
        num_limbus_lifting = \
            self.cov_calc.derivative_calculator_limbus_lifting.\
            get_num_parameters()

        num_rows = \
            num_eye_shape + \
            num_eye_pose + \
            1 * (num_pupil_lifting + num_limbus_lifting)  # 1D.

        num_leds = \
            self.cov_calc.derivative_calculator_leds.get_num_parameters()
        num_extrinsics = \
            self.cov_calc.derivative_calculator_camera_extrinsics.\
            get_num_parameters()
        num_intrinsics = \
            self.cov_calc.derivative_calculator_camera_intrinsics.\
            get_num_parameters()
        num_glints = self.cov_calc.derivative_data.num_visible_glints

        num_cols = \
            num_leds + \
            num_extrinsics + \
            num_intrinsics + \
            2 * (num_glints + num_pupil_lifting + num_limbus_lifting)  # 2D

        self.assertTrue(d_optim_d_data.shape[0] == num_rows)
        self.assertTrue(d_optim_d_data.shape[1] == num_cols)

        # Reset derivatives, start again.
        self.cov_calc.reset_derivatives()
        list_gaze_rotation_deg = \
            [torch.tensor([15.0, -10.0]), torch.tensor([-20.0, 15.0])]
        self.cov_calc.set_list_of_gaze_angles(list_gaze_rotation_deg)
        d_optim_d_data = self.cov_calc.compute_d_optim_d_data()

        # New number for pupil and limbus lifting. This works because the list
        # of gaze direction has only two terms, and the new value will
        # correspond to the number of parameters of the last gaze direction.
        num_pupil_lifting = \
            self.cov_calc.derivative_calculator_pupil_lifting.\
            get_num_parameters()
        num_limbus_lifting = \
            self.cov_calc.derivative_calculator_limbus_lifting.\
            get_num_parameters()
        # New number of glints. Also corresponding to the last gaze direction.
        num_glints = self.cov_calc.derivative_data.num_visible_glints

        # We get one more eye pose and one more set of lifting parameters. We
        # do not get more eye-shape parameters, as the shape of the eyes is
        # fixed for all gaze directions.
        num_rows += num_eye_pose + 1 * (num_pupil_lifting + num_limbus_lifting)

        # We get one more set of image measurements.
        num_cols += 2 * (num_glints + num_pupil_lifting + num_limbus_lifting)

        self.assertTrue(d_optim_d_data.shape[0] == num_rows)
        self.assertTrue(d_optim_d_data.shape[1] == num_cols)

    def test_compute_covariance(self):
        """Test computation of covariance of nose in eye-shape parameters.
        """

        # Create a list of gaze directions.
        list_gaze_rotation_deg = [torch.tensor([15.0, -10.0]), ]
        self.cov_calc.set_list_of_gaze_angles(list_gaze_rotation_deg)
        cov_eye_shape = self.cov_calc.compute_covariance()

        num_shape = \
            self.cov_calc.derivative_calculator_eye_shape.get_num_parameters()
        self.assertTrue(cov_eye_shape.shape[0] == num_shape)  # type: ignore

        # Is valid covariance? Need to be forgiving, as some eigenvalues may be
        # small negative numbers.
        self.assertTrue(
            sensitivity_analysis.is_valid_covariance(cov_eye_shape)
        )

        # Reset derivatives, start again.
        self.cov_calc.reset_derivatives()
        list_gaze_rotation_deg = \
            [torch.tensor([15.0, -10.0]), torch.tensor([-20.0, 15.0])]
        self.cov_calc.set_list_of_gaze_angles(list_gaze_rotation_deg)
        cov_eye_shape = self.cov_calc.compute_covariance()

        num_shape = \
            self.cov_calc.derivative_calculator_eye_shape.get_num_parameters()
        self.assertTrue(cov_eye_shape.shape[0] == num_shape)  # type: ignore

        # Is valid covariance.
        self.assertTrue(
            sensitivity_analysis.is_valid_covariance(cov_eye_shape)
        )

    def test_save_covariance(self):
        """Test saving of covariance matrix.
        """

        # Create a list of gaze directions.
        list_gaze_rotation_deg = [torch.tensor([15.0, -10.0]), ]
        self.cov_calc.set_list_of_gaze_angles(list_gaze_rotation_deg)
        self.cov_calc.compute_covariance()
        parameter_file_name = os.path.join(
            sensitivity_analysis_tests_configs.SENSITIVITY_ANALYSIS_TEST_DIR,
            r"sensitivity_analysis_tests_data" +
            r"/default_eye_shape_covariances.json"
        )
        self.cov_calc.save_covariance(parameter_file_name=parameter_file_name)

        # Load saved covariance matrix. The set method is defined in the base
        # class EyePoseCovariances, and it sets the member variable
        # self.eye_shape_covariance
        self.cov_calc.load_eye_shape_covariance(parameter_file_name)

        self.assertTrue(
            torch.allclose(
                self.cov_calc.covariance,
                self.cov_calc.eye_shape_covariance
            )
        )


if __name__ == "__main__":
    unittest.main()
