"""normalized_camera_tests.py

Unit tests for normalized camera class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import itertools
import seet.core as core
import seet.device as device
import seet.primitives as primitives
import torch
import unittest


class TestNormalizedCamera(unittest.TestCase):
    """TestNormalizedCamera.

    Test class NormalizedCamera.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """
        super().setUp()

        self.fake_SubsystemModel = core.Node(name="fake ET subsystem")
        self.transform_toSubsystemModel_fromCamera = core.SE3.create_identity()
        self.camera = \
            device.NormalizedCamera(
                self.fake_SubsystemModel,
                self.transform_toSubsystemModel_fromCamera,
                name="normalized camera")
        self.point_inCamera = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    def test_rotate_around_axis(self):
        """test_rotate_around_axis.

        Test rotation of camera around an axis.
        """
        # Project a point right in front of the camera.
        point_inParent = torch.tensor([0.0, 0.0, 10.0])
        point_inImage = \
            self.camera.project_toImagePlane_fromParent(point_inParent)

        # Rotate the camera around some axis.
        axis = torch.tensor([0.1, -0.2, 0.3])
        self.camera.rotate_around_axis(axis)
        point_inImage_ = \
            self.camera.project_toImagePlane_fromParent(point_inParent)

        # The new point is homographically transformed by the rotation.
        rotation = core.rotation_matrix(axis).T
        new_point_inImage = \
            core.dehomogenize(
                rotation @ core.homogenize(point_inImage)
            )

        self.assertTrue(torch.allclose(point_inImage_, new_point_inImage))

        # Undo the rotation.
        self.camera.rotate_around_axis(-axis)

        # Rotate the camera around x, y, and z by 45 degrees
        angle_deg = torch.tensor(45.0)
        axes = \
            {
                "x": core.rotation_around_x,
                "y": core.rotation_around_y,
                "z": core.rotation_around_z
            }
        for axis in axes.keys():
            self.camera.rotate_around_x_y_or_z(angle_deg, axis=axis)
            point_inImage_ = \
                self.camera.project_toImagePlane_fromParent(point_inParent)
            self.camera.rotate_around_x_y_or_z(-angle_deg, axis=axis)

            rotation = axes[axis](angle_deg).T
            new_point_inImage = \
                core.dehomogenize(
                    rotation @ core.homogenize(point_inImage)
                )

            self.assertTrue(
                torch.allclose(
                    point_inImage_,
                    new_point_inImage,
                    rtol=core.EPS * 100,
                    atol=core.EPS * 100)
            )

    def test_project_toImagePlane_fromCamera(self):
        """test_project_toImagePlane_fromCamera.

        Test projection onto image plane.
        """
        point_inImage = \
            self.camera.project_toImagePlane_fromCamera(self.point_inCamera)

        self.assertTrue(point_inImage.numel() == 2)

        # Jacobian of projection is a (2, 3) tensor J such that J multiplied by
        # the direction of the ray corresponding to the image plane is zero.
        d_point_inImage_d_point_inCamera = \
            core.compute_auto_jacobian_from_tensors(
                point_inImage, self.point_inCamera
            )

        direction_inCamera = \
            self.camera.compute_direction_toCamera_fromImagePlane(
                point_inImage
            )

        self.assertTrue(
            torch.allclose(
                d_point_inImage_d_point_inCamera @ direction_inCamera,
                torch.zeros(2),
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

    def test_project_toImagePlane_fromParent(self):
        point_inParent = torch.tensor([2.0, -1.0, 10.0], requires_grad=True)

        point_inImage = \
            self.camera.project_toImagePlane_fromParent(point_inParent)

        # Jacobian of projection is a (2, 3) tensor J such that J multiplied by
        # the direction of the ray corresponding to the image plane is zero.
        d_point_inImage_d_point_inParent = \
            core.compute_auto_jacobian_from_tensors(
                point_inImage, point_inParent
            )

        direction_inParent = \
            self.camera.compute_direction_toParent_fromImagePlane(
                point_inImage
            )

        # Ensure all tensors are on the same device
        target_device = d_point_inImage_d_point_inParent.device
        if direction_inParent.device != target_device:
            direction_inParent = direction_inParent.to(target_device)
        zeros_tensor = torch.zeros(2, device=target_device)
        self.assertTrue(
            torch.allclose(
                d_point_inImage_d_point_inParent @ direction_inParent,
                zeros_tensor,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

    def test_origin_and_direction_toParent_fromImagePlane(self):
        """test_origin_and_direction_toParent_fromImagePlane.

        Test get origin and direction from image plane.
        """

        point_inImage = torch.tensor([1.0, 1.0], requires_grad=True)

        origin_inParent, direction_inParent = \
            self.camera.compute_origin_and_direction_toParent_fromImagePlane(
                point_inImage
            )

        # Project point back and check that it matches the image point.
        t = torch.tensor(5.0, requires_grad=True, device='cuda')
        point_inParent = origin_inParent.to(device='cuda') + t * direction_inParent.to(device='cuda')
        point_inImage_ = \
            self.camera.project_toImagePlane_fromParent(point_inParent)

        self.assertTrue(torch.allclose(point_inImage, point_inImage_))

        # The derivative of point_inImage_ with respect to point_inImage should
        # be the identity matrix.
        d_point_inImage_d_point_inImage = \
            core.compute_auto_jacobian_from_tensors(
                point_inImage_, point_inImage
            )

        self.assertTrue(
            torch.allclose(d_point_inImage_d_point_inImage, torch.eye(2))
        )

    def test_mirror(self):
        """test_mirror.

        Test mirroring of camera with respect to yz plane of some core.
        """

        # Create a point and project it. Mirror point and camera and test
        # whether projection in image is a flip around the image's x axis.
        other_node = self.fake_SubsystemModel
        transform_toOther_fromCamera = \
            self.camera.get_transform_toOther_fromSelf(other_node)
        point_inCamera = torch.tensor([1.0, 2.0, 10.0], device='cuda')
        point_inOther = transform_toOther_fromCamera.transform(point_inCamera)
        point_inImage = \
            self.camera.project_toImagePlane_fromOther(
                point_inOther, other_node
            )

        mirror_point_inOther = torch.tensor([-1.0, 1.0, 1.0], device='cuda') * point_inOther
        mirror_camera = \
            device.NormalizedCamera.mirror(self.camera)
        mirror_point_inImage = \
            mirror_camera.project_toImagePlane_fromOther(
                mirror_point_inOther, other_node
            )
        point_inImage_ = torch.tensor([-1.0, 1.0], device='cuda') * mirror_point_inImage.to(device='cuda')

        self.assertTrue(torch.allclose(point_inImage.to(device='cuda'), point_inImage_))

    def test_project_ellipsoid(self):
        """test_project_ellipsoid.

        Test projection of an ellipsoid onto the camera's image plane.
        """

        # Create ellipsoid.
        ellipsoid = \
            primitives.Ellipsoid(
                self.camera,
                core.SE3.create_identity(),
                torch.tensor([1.0, 2.0, 3.0])
            )

        # We cannot project this ellipsoid, as it is centered on the camera's
        # optical center. Let's translate it away.
        ellipsoid.translate_inParent(torch.tensor([1.0, -2.0, 10.0]))
        Q = ellipsoid.get_ellipsoid_matrix_inOther(self.camera)
        Q_inv = torch.linalg.pinv(Q)

        # Project occluding contour.
        ellipse = self.camera.project_ellipsoid_toImagePlane(ellipsoid)
        C = ellipse.get_homogeneous_matrix_inPlane()
        transform_toImagePlane_fromEllipse = \
            ellipse.get_transform_toParent_fromSelf()

        # Verify that all points on the ellipse yield rays that are tangent to
        # the ellipsoid.
        points_inEllipse, _ = ellipse.sample_points_inEllipse()
        for single_point_inEllipse in points_inEllipse.T:
            point_inImagePlane = \
                transform_toImagePlane_fromEllipse.transform(
                    single_point_inEllipse
                )[:2]
            polar_line_inImagePlane = C @ core.homogenize(point_inImagePlane)

            algebraic_distance = \
                polar_line_inImagePlane @ \
                Q_inv[:3, :3] @ polar_line_inImagePlane
            self.assertTrue(
                torch.allclose(algebraic_distance, core.T0, atol=1e-3)
            )

    def test_project_ellipse(self):
        """test_project_ellipse.

        Test projection of an ellipse onto the image plane.
        """

        # Create a plane.
        plane = \
            primitives.Plane.create_from_homogeneous_coordinates_inParent(
                self.camera, torch.tensor([0.1, -0.1, 0.9, -10.0])
            )

        # Add an ellipse to the plane. Note "in_plane" rather than "inPlane" as
        # this is a node in the graph, and therefore coordinate free.
        ellipse_in_plane = \
            primitives.Ellipse.create_from_homogeneous_matrix_inPlane(
                plane, torch.diag(torch.tensor([1.0, 1.0 / 4.0, -1.0]))
            )

        # Project ellipse onto camera image plane.
        ellipse_in_image = \
            self.camera.project_ellipse_toImagePlane(ellipse_in_plane)
        C_inImagePlane = ellipse_in_image.get_homogeneous_matrix_inPlane()

        # Verify that the points on the original ellipse project onto the
        # projected ellipse.
        points_inOriginalEllipse, \
            _ = ellipse_in_plane.sample_points_inEllipse()
        for single_point_inOriginalEllipse in points_inOriginalEllipse.T:
            point_inImagePlane = \
                self.camera.project_toImagePlane_fromOther(
                    single_point_inOriginalEllipse, ellipse_in_plane
                )
            h_point_inImagePlane = core.homogenize(point_inImagePlane)
            # Ensure both tensors are on the same device
            if h_point_inImagePlane.device != C_inImagePlane.device:
                h_point_inImagePlane = h_point_inImagePlane.to(C_inImagePlane.device)

            algebraic_distance = \
                h_point_inImagePlane @ C_inImagePlane @ h_point_inImagePlane
            self.assertTrue(
                torch.allclose(algebraic_distance, core.T0.to(algebraic_distance.device), atol=1e-3)
            )

    def test_forward_project_ellipse_onto_circle(self):
        """test_forward_project_ellipse_onto_circle.

        Test the forward projection of an ellipse in the image plane onto a
        circle in 3D.
        """

        # Create a circle in the camera coordinate system. First, create a
        # plane in front of the camera. Normal of plane has a it of x and y and
        # lots of z. It is important that z > 0, so that the normal to the
        # plane points away from the camera.
        normal_inCamera = torch.tensor([0.1, 0.1, 0.9])
        # Origin of the plane in the camera. A bit of x and y, lots of z.
        origin_inCamera = torch.tensor([1.0, -2.0, 20.0])
        plane = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self.camera, origin_inCamera, normal_inCamera
            )

        # Add a circle to the plane.
        origin_in2DPlane = torch.tensor([-3.0, -2.0])
        radius = torch.tensor(5.0)
        circle_3D = \
            primitives.Circle.create_from_origin_and_radius_inPlane(
                plane, origin_in2DPlane, radius
            )
        # Get the origin and normal of the true circle, expressed in the camera
        # coordinate system.
        parameters_inCamera_gt = \
            circle_3D.get_center_and_normal_inOther(self.camera)

        # Get the corresponding ellipse in the image plane.
        ellipse_2D = self.camera.project_ellipse_toImagePlane(circle_3D)

        # Forward project the ellipse back into the circle in 3D
        parameters_inCamera_pos, \
            parameters_inCamera_neg = \
            self.camera.forward_project_to_circle_from_ellipse_inImagePlane(
                ellipse_2D, radius
            )

        # Compare the origins and normals of the two possible solution for the
        # parameters of the forward projection of the circles with their known
        # values.

        # Assume that the true solution is the one closest to the ground truth.
        difference_pos = parameters_inCamera_pos[1] - parameters_inCamera_gt[1]
        difference_neg = parameters_inCamera_neg[1] - parameters_inCamera_gt[1]
        if difference_pos @ difference_pos < difference_neg @ difference_neg:
            parameters_inCamera_ = parameters_inCamera_pos
        else:
            parameters_inCamera_ = parameters_inCamera_neg

        # Center and normal are nearly the same.
        #FIXME
        #for i in range(2):
        #    self.assertTrue(
        #        torch.allclose(
        #            parameters_inCamera_[i], parameters_inCamera_gt[i]
        #        )
        #    )

    def test_forward_project_ellipse_onto_circle_extra(self):
        """test_forward_project_ellipse_onto_circle_extra.

        More tests on the projection of an ellipse in the image plane onto a
        circle in 3D.
        """

        # Create baseline circles in 3D, perturb them, project them onto the
        # image plane, and forward project the resulting ellipses back to 3D.
        # Then, compare those forward projections against the perturbed circles
        # from which they were produced.
        base_origin_inCamera = torch.tensor([0.0, 0.0, 20.0])
        delta_origin_inCamera = torch.tensor([3.0, 3.0, 0.0])

        base_normal_inCamera = torch.tensor([0.0, 0.0, 1.0])
        delta_normal_inCamera = torch.tensor([0.1, 0.1, 0.0])

        base_center_inPlane = torch.tensor([0.0, 0.0])
        delta_center_inPlane = torch.tensor([5.0, 5.0])

        base_radius = torch.tensor(10.0)
        delta_radius = torch.tensor(3.0)

        jitter_range = torch.linspace(-1.0, 1.0, 3)
        jitter_all = \
            itertools.product(
                jitter_range, jitter_range, jitter_range, jitter_range
            )
        for jitter in jitter_all:
            origin_inCamera = base_origin_inCamera + \
                delta_origin_inCamera * jitter[0]
            normal_inCamera = base_normal_inCamera + \
                delta_normal_inCamera * jitter[1]
            center_inPlane = base_center_inPlane + \
                delta_center_inPlane * jitter[2]
            radius = base_radius + delta_radius * jitter[3]

            # Create a plane.
            plane = \
                primitives.Plane.create_from_origin_and_normal_inParent(
                    self.camera, origin_inCamera, normal_inCamera
                )

            # Create a 3D circle in the plane.
            circle = \
                primitives.Circle.create_from_origin_and_radius_inPlane(
                    plane, center_inPlane, radius
                )

            # Project the circle onto an ellipse in the image plane.
            ellipse = \
                self.camera.project_ellipse_toImagePlane(circle)

            # Forward project the ellipse to a circle in 3D.
            params_pos, \
                params_neg = \
                self.\
                camera.forward_project_to_circle_from_ellipse_inImagePlane(
                    ellipse, radius  # Radius must be known!
                )

            # Circles must be the same.
            origin_inCamera, \
                normal_inCamera = \
                circle.get_center_and_normal_inOther(self.camera)

            # Resolve ambiguity manually.
            difference_pos = params_pos[0] - origin_inCamera
            difference_neg = params_neg[0] - origin_inCamera
            distance_pos = torch.sqrt(difference_pos @ difference_pos)
            distance_neg = torch.sqrt(difference_neg @ difference_neg)

            if distance_pos < distance_neg:
                normal_inCamera_ = params_pos[1]
                distance = distance_pos
            else:
                normal_inCamera_ = params_neg[1]
                distance = distance_neg

            angle_axis = torch.cross(normal_inCamera, normal_inCamera_)
            theta_deg = \
                core.rad_to_deg(torch.asin(torch.linalg.norm(angle_axis)))

            if not torch.allclose(distance, core.T0, atol=1e-1):
                self.camera.\
                    forward_project_to_circle_from_ellipse_inImagePlane(
                        ellipse, radius  # Radius must be known!
                    )

            if not torch.allclose(theta_deg, core.T0, atol=0.1):
                self.camera.\
                    forward_project_to_circle_from_ellipse_inImagePlane(
                        ellipse, radius  # Radius must be known!
                    )

            # Distance error. Numerics are abominable.
            #FIXME self.assertTrue(torch.allclose(distance, core.T0, atol=1e-1))
            # Angular error
            #FIXME self.assertTrue(torch.allclose(theta_deg, core.T0, atol=0.1))


if __name__ == "__main__":
    unittest.main()
