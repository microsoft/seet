"""Tests for sensitivity analysis of camera intrinsic parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.sensitivity_analysis import CameraIntrinsicsDerivatives
from tests.sensitivity_analysis_tests import test_utils
import torch
import unittest


class TestCameraIntrinsicsDerivatives(test_utils.TestCommonUtils):
    """Unit tests for sensitivity analysis of camera intrinsic parameters.
    """

    def setUp(self):
        """setUp.

        Generate data for tests.
        """

        super().setUp()

        self.extraSetup(CameraIntrinsicsDerivatives)
        self.M = 9  # There are 9 intrinsic camera parameters.

    def test_size_compute_d_glints_d_parameters(self):
        """Test derivatives of glints with respect to intrinsic parameters.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_glints_inPixels,
            self.derivative_calculator.compute_d_glints_d_parameters,
            self.M
        )

    def test_size_compute_d_pupil_d_parameters(self):
        """Test derivatives of pupil with respect to intrinsic parameters.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.generate_pupil_inPixels,
            self.derivative_calculator.compute_d_pupil_d_parameters,
            self.M
        )

    def test_size_compute_d_limbus_d_parameters(self):
        """Test derivatives of limbus with respect to intrinsic parameters.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_limbus_inPixels,
            self.derivative_calculator.compute_d_limbus_d_parameters,
            self.M
        )

    def test_compute_d_glints_d_pinhole_intrinsics(self):
        """Test derivatives of glints with respect to pinhole intrinsics.
        """

        # Points in pixels are given by
        #
        # pt_pix = F @ distorted(pt_im) + [px, py].
        #
        # Therefore,
        #
        # d_pt_pix_d_fx_fy = diag(distorted(pt_im)),
        #
        # d_pt_pix_d_px_py = eye(2)

        derivative_calculator = self.derivative_calculator
        derivative_data = derivative_calculator.derivative_data
        camera = derivative_data.camera

        derivative_data.generate_glints_inPixels()
        all_glints_inPixels = derivative_data.all_glints_inPixels

        d_all_glints_d_intrinsics = \
            derivative_calculator.compute_d_glints_d_parameters()

        for glint_inPixels, d_glint_d_intrinsics in zip(
            all_glints_inPixels, d_all_glints_d_intrinsics
        ):
            d_glint_d_focal_length = d_glint_d_intrinsics[:, :2]

            glint_inImagePlane = camera.get_point_inImagePlane(glint_inPixels)
            distorted_glint_inImagePlane = \
                camera.distort_point_inUndistortedImage(glint_inImagePlane)

            self.assertTrue(
                torch.allclose(
                    d_glint_d_focal_length,
                    torch.diag(distorted_glint_inImagePlane),
                    atol=1e-7
                )
            )

            d_glint_d_principal_point = d_glint_d_intrinsics[:, 2:4]

            self.assertTrue(
                torch.allclose(
                    d_glint_d_principal_point, torch.eye(2)
                )
            )


if __name__ == "__main__":
    unittest.main()
