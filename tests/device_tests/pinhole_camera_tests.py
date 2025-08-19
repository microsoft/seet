"""pinhole_camera_tests.py

Unit tests for pinhole camera class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.device as device
from tests.device_tests import normalized_camera_tests
import torch
import unittest


class TestPinholeCamera(normalized_camera_tests.TestNormalizedCamera):
    """TestPinholeCamera.

    Test class PinholeCamera.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """
        super().setUp()

        self.fake_SubsystemModel = core.Node(name="fake ET subsystem")
        self.transform_toSubsystemModel_fromCamera = core.SE3.create_identity()
        self.camera = \
            device.PinholeCamera(
                self.fake_SubsystemModel,
                self.transform_toSubsystemModel_fromCamera,
                requires_grad=True
            )
        self.point_in3D = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    def test_set_pinhole_intrinsics(self):
        """test_set_pinhole_intrinsics.

        Test setting of pinhole intrinsics.
        """
        # Focal length is a scale factor. Therefore, the derivative of a point
        # in pixel coordinates with respect to focal length is equal to the
        # diagonal matrix built from the normalized coordinates of that point.
        for fun \
                in [
                    self.camera.project_toImagePlane_fromCamera,
                    self.camera.project_toImagePlane_fromParent
                ]:
            point_inImage = fun(self.point_in3D)
            point_inPixels = self.camera.get_point_inPixels(point_inImage)

            d_point_inPixels_d_focal_lengths = \
                core.compute_auto_jacobian_from_tensors(
                    point_inPixels, self.camera.focal_lengths
                )

            self.assertTrue(
                torch.allclose(
                    d_point_inPixels_d_focal_lengths, torch.diag(point_inImage)
                )
            )

            # The derivative of the point in pixel coordinates with respect to
            # the coordinates of the principal point is the (2, 2) identity
            # matrix.
            d_point_inPixels_d_principal_point = \
                core.compute_auto_jacobian_from_tensors(
                    point_inPixels, self.camera.principal_point
                )

            self.assertTrue(
                torch.allclose(
                    d_point_inPixels_d_principal_point, torch.eye(2)
                )
            )

    def test_projection_toPixels(self):
        """test_projection_toPixels.

        Test projection of points to pixel coordinates.
        """
        for fun \
                in [
                    self.camera.project_toImagePlane_fromCamera,
                    self.camera.project_toImagePlane_fromParent
                ]:
            point_inImage = fun(self.point_in3D)
            point_inPixels = self.camera.get_point_inPixels(point_inImage)

            point_inPixels_ = \
                self.camera.project_toPixels_fromCamera(self.point_in3D)

            self.assertTrue(torch.allclose(point_inPixels, point_inPixels_))

        # Derivative of projection with respect to point in 3D must be zero
        # along viewing ray.
        point_inPixels = \
            self.camera.project_toPixels_fromCamera(self.point_in3D)
        direction_in3D = \
            self.camera.compute_direction_toCamera_fromPixels(point_inPixels)
        d_point_inPixels_d_point_in3D = \
            core.compute_auto_jacobian_from_tensors(
                point_inPixels, self.point_in3D
            )

        self.assertTrue(
            torch.allclose(
                d_point_inPixels_d_point_in3D @ direction_in3D,
                torch.zeros(2),
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

    def test_is_point_inCamera_in_field_of_view(self):
        """test_is_point_inCamera_in_field_of_view.

        Test check of whether point is in the camera's field of view.
        """
        # Generate points in the camera's field of view.
        for point_inPixels in (
            torch.tensor([-0.4, -0.4]),
            torch.tensor([399.4, 399.4]),
            torch.tensor([-0.4, 399.4]),
            torch.tensor([399.4, -0.4])
        ):
            origin_inParent, direction_inParent = \
                self.camera.compute_origin_and_direction_inParent_fromPixels(
                    point_inPixels
                )
            point_in_field_of_view_inParent = \
                origin_inParent + 10 * direction_inParent

            self.assertTrue(
                self.camera.is_point_inParent_in_field_of_view(
                    point_in_field_of_view_inParent
                )
            )

        # Generate points outside the camera's field of view.
        for point_inPixels in (
            torch.tensor([-0.6, -0.6]),
            torch.tensor([399.7, 399.6]),
            torch.tensor([-0.6, 399.6]),
            torch.tensor([399.6, -0.6])
        ):
            origin_inParent, direction_inParent = \
                self.camera.compute_origin_and_direction_inParent_fromPixels(
                    point_inPixels
                )
            point_in_field_of_view_inParent = \
                origin_inParent + 10 * direction_inParent

            self.assertFalse(
                self.camera.is_point_inParent_in_field_of_view(
                    point_in_field_of_view_inParent
                )
            )

    def test_mirror(self):
        """test_mirror.

        Test mirroring of pinhole camera.
        """

        # Mirroring of normalized camera superclass.
        super().test_mirror()

        # Mirroring happens in the normalized image plane. Its effect on the
        # pixel values is different.
        point_inPixels = \
            self.camera.project_toPixels_fromCamera(self.point_in3D)

        mirror_camera = device.PinholeCamera.mirror(self.camera)

        mirror_point_in3D = torch.tensor([-1.0, 1.0, 1.0]) * self.point_in3D
        mirror_point_inPixels = \
            mirror_camera.project_toPixels_fromCamera(mirror_point_in3D)
        mirror_point_inImage = \
            mirror_camera.get_point_inImagePlane(mirror_point_inPixels)
        point_inImage_ = torch.tensor([-1.0, 1.0]) * mirror_point_inImage
        point_inPixels_ = \
            mirror_camera.get_point_inPixels(point_inImage_)

        self.assertTrue(torch.allclose(point_inPixels, point_inPixels_))


if __name__ == "__main__":
    unittest.main()
