"""polynomial3K_camera_tests.py

Unit tests for polynomial 3K camera class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.device as device
from tests.device_tests import device_tests_configs
from tests.device_tests import pinhole_camera_tests
import os
import torch
import unittest


class TestPolynomial3KCamera(pinhole_camera_tests.TestPinholeCamera):
    """TestPolynomial3KCamera.

    Test class Polynomial3KCamera.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.

        This implicitly tests the __init__ method of the class
        Polynomial3KCamera.
        """
        super().setUp()

        # Replace the pinhole camera with a polynomial camera.
        self.camera = \
            device.Polynomial3KCamera(
                self.fake_SubsystemModel,
                self.transform_toSubsystemModel_fromCamera,
                requires_grad=True
            )

    def test_set_distortion_parameters(self):
        """test_set_distortion_parameters.

        Test setting of distortion parameters.
        """

        distortion_center = self.camera.distortion_center
        distortion_coefficients = self.camera.distortion_coefficients

        # Coefficients are differentiable.
        self.assertTrue(distortion_center.requires_grad)
        self.assertTrue(distortion_coefficients.requires_grad)

        # Setting with None should not change parameters.
        self.camera.set_distortion_parameters()
        new_distortion_center = self.camera.distortion_center
        new_distortion_coefficients = self.camera.distortion_coefficients

        self.assertTrue(
            torch.allclose(distortion_center, new_distortion_center)
        )
        self.assertTrue(
            torch.allclose(
                distortion_coefficients, new_distortion_coefficients
            )
        )

        # Coefficients are still differentiable.
        self.assertTrue(new_distortion_center.requires_grad)
        self.assertTrue(new_distortion_coefficients.requires_grad)

        # Setting to some arbitrary values.
        distortion_center_ = torch.tensor([0.01, -0.02])
        distortion_coefficients_ = torch.tensor([-0.11, 0.22, -0.33])
        self.camera.set_distortion_parameters(
            distortion_center=distortion_center_,
            distortion_coefficients=distortion_coefficients_
        )
        distortion_center = self.camera.distortion_center
        distortion_coefficients = self.camera.distortion_coefficients

        self.assertTrue(
            torch.allclose(distortion_center, distortion_center_)
        )
        self.assertTrue(
            torch.allclose(
                distortion_coefficients, distortion_coefficients_
            )
        )

        # Coefficients are no longer differentiable.
        self.assertFalse(distortion_center.requires_grad)
        self.assertFalse(distortion_coefficients.requires_grad)

    def test_compute_radial_distortion(self):
        """test_compute_radial_distortion.

        Test computation of radial factor applied to decentered points.
        """
        distortion_coefficients = \
            torch.tensor([1.00, -0.10, 0.01], requires_grad=True)
        radial_distortion = \
            device.Polynomial3KCamera.compute_radial_distortion(
                torch.zeros(2), distortion_coefficients
            )

        # Distortion at center of distortion is zero.
        self.assertTrue(torch.allclose(radial_distortion, core.T1))

        # Derivative with respect to distortion coefficients at center of
        # distortion is zero.
        d_radial_distortion_d_distortion_coefficients = \
            core.compute_auto_jacobian_from_tensors(
                radial_distortion, distortion_coefficients
            )

        self.assertTrue(
            torch.allclose(
                d_radial_distortion_d_distortion_coefficients,
                torch.zeros_like(d_radial_distortion_d_distortion_coefficients)
            )
        )

        decentered_point_inUndistortedImage = \
            torch.tensor([0.01, -0.01], requires_grad=True)
        radial_distortion = \
            device.Polynomial3KCamera.compute_radial_distortion(
                decentered_point_inUndistortedImage, distortion_coefficients
            )
        norm = torch.linalg.norm(decentered_point_inUndistortedImage)
        radial_distortion_ = \
            core.T1 + \
            distortion_coefficients[0] * norm**2 + \
            distortion_coefficients[1] * norm**4 + \
            distortion_coefficients[2] * norm**6

        self.assertTrue(torch.allclose(radial_distortion, radial_distortion_))

        d_radial_distortion_d_distortion_coefficients = \
            core.compute_auto_jacobian_from_tensors(
                radial_distortion, distortion_coefficients
            )
        d_radial_distortion_d_distortion_coefficients_ = \
            torch.stack((norm**2, norm**4, norm**6))

        self.assertTrue(
            torch.allclose(
                d_radial_distortion_d_distortion_coefficients,
                d_radial_distortion_d_distortion_coefficients_
            )
        )

        d_radial_distortion_d_decentered_point = \
            core.compute_auto_jacobian_from_tensors(
                radial_distortion, decentered_point_inUndistortedImage
            )
        d_radial_distortion_d_decentered_point_ = \
            (
                distortion_coefficients @
                torch.stack((2 * core.T1, 4 * norm**2, 6 * norm**4))
            ).sum() * decentered_point_inUndistortedImage

        self.assertTrue(
            torch.allclose(
                d_radial_distortion_d_decentered_point,
                d_radial_distortion_d_decentered_point_
            )
        )

    def test_distort_undistort(self):
        """test_distort_undistort.

        Test distortion and un-distortion of points.
        """

        # Setting to some arbitrary values.
        distortion_center = torch.tensor([0.01, -0.02])
        distortion_coefficients = torch.tensor([-0.003, 0.002, -0.001])

        resolution = self.camera.resolution.clone().detach().tolist()
        self.camera.set_distortion_parameters(
            distortion_center=distortion_center,
            distortion_coefficients=distortion_coefficients
        )

        # These points are assumed to have undergone distortion after
        # projection. Only then intrinsic parameters apply.
        for point_inPixels in (
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, resolution[1] - 1]),
            torch.tensor([resolution[0] - 1, 0.0]),
            torch.tensor([resolution[0] - 1, resolution[1] - 1])
        ):
            point_inImage = self.camera.get_point_inImagePlane(point_inPixels)
            point_inPixels_ = self.camera.get_point_inPixels(point_inImage)

            self.assertTrue(
                torch.allclose(
                    point_inPixels,
                    point_inPixels_,
                    rtol=core.EPS * 1000,
                    atol=core.EPS * 1000)
            )


if __name__ == "__main__":
    unittest.main()
