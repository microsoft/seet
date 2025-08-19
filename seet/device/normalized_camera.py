"""normalized_camera.py.

Class representing a normalized camera. A normalized camera is a pinhole
camera without intrinsic parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
import torch


class NormalizedCamera(core.Node):
    """NormalizedCamera.

    Class for a normalized pinhole camera. A normalized camera consists of a
    point in the  pose graph corresponding to its optical center and an image
    the plane.

    The origin of the camera coordinate system is the camera optical center,
    and its orientation is naturally defined by its optical axis and the
    orientation of its image plane.
    """

    def __init__(
        self,
        subsystem_model,
        transform_toSubsystemModel_fromCamera,
        name="",
        requires_grad=False
    ):
        """
        Initialize a normalized-camera object.

        Args:
            subsystem_model (SubsystemModel): SubsystemModel object of which
            camera is a child node.

            transform_toSubsystemModel_fromCamera (core.SE3): element of
            SE(3) corresponding to the transformation from the coordinate
            system of the camera to that of the eye-tracking subsystem to which
            the camera is attached.

            name (str, optional): name of camera. Defaults to "".

        Returns:
            NormalizedCamera: normalized-camera object.
        """

        super().__init__(
            subsystem_model,
            transform_toSubsystemModel_fromCamera,
            name=name,
            requires_grad=requires_grad
        )

        self.optical_center = \
            primitives.Point(
                self,
                core.SE3.create_identity(),  # type: ignore
                name="optical center of camera " + name
            )

        image_plane_origin_inCamera = torch.tensor([0.0, 0.0, 1.0])
        image_plane_normal_inCamera = torch.tensor([0.0, 0.0, 1.0])
        self.image_plane = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self,
                image_plane_origin_inCamera,
                image_plane_normal_inCamera,
                name="image plane of camera " + name
            )

    def rotate_around_axis(self, angle_axis):
        """rotate_around_axis.

        Rotate the camera around the axis with direction angle_axis with
        magnitude equal to the norm of angle_axis.

        Args:
            axis (torch.Tensor): (3,) tensor representing the angle-axis in the
            camera's coordinate system around which to rotate the camera.
        """
        rotation = core.rotation_matrix(angle_axis)
        rotation_toSelf_fromSelf = core.SO3(rotation)
        self.transform_toParent_fromSelf = \
            core.SE3.compose_transforms(
                self.transform_toParent_fromSelf, rotation_toSelf_fromSelf
            )

    def rotate_around_x_y_or_z(self, angle_deg, axis="x"):
        """rotate_around_x_y_or_z.

        Rotate the camera around the camera's x, y, or z axis by the given
        angle in degrees.

        Args:
            axis (str, optional): "x", "y", or "z", indicating around which
            axis to rotate. Defaults to "x".

            angle_deg (float or torch.float): angle in degrees by which to
            rotate the camera.
        """
        normalized_axis = torch.tensor([1.0, 0.0, 0.0])
        if axis == "y":
            normalized_axis = torch.tensor([0.0, 1.0, 0.0])
        elif axis == "z":
            normalized_axis = torch.tensor([0.0, 0.0, 1.0])

        angle_axis = normalized_axis * core.deg_to_rad(angle_deg)

        self.rotate_around_axis(angle_axis)

    def project_toImagePlane_fromCamera(self, point_inCamera):
        """project_toImagePlane_fromCamera.

        Project a point in the camera coordinate system onto the camera's
        image plane.

        Args:
            point_inCamera (torch.Tensor): (3,) tensor corresponding to
            coordinates of a point in the camera's coordinate system.

        Returns:
            torch.Tensor: (2,) tensor corresponding to coordinates of
            projection in the coordinates of the image plane.
        """
        intersection_inCamera = \
            self.image_plane.intersect_from_origin_and_direction_inParent(
                point_inCamera, -point_inCamera  # Always points to (0, 0, 0)
            )

        # We return the normalized coordinates in the image plane. The third
        # coordinate is always 1, and we don't care about it.
        return intersection_inCamera[:2]

    def project_toImagePlane_fromParent(self, point_inParent):
        """project_toImagePlane_fromParent.

        Project a point in the coordinate system of the camera's SubsystemModel
        parent node onto the camera's image plane.

        Args:
            point_inParent (torch.Tensor): (3,) tensor corresponding to
            coordinates of a point in the camera's SubsystemModel parent node.

        Returns:
            torch.Tensor: (2,) tensor corresponding to coordinates of
            projection in the coordinates of the image plane.
        """
        point_inCamera = \
            self.transform_toParent_fromSelf.inverse_transform(point_inParent)

        return self.project_toImagePlane_fromCamera(point_inCamera)

    def project_toImagePlane_fromOther(self, point_inOther, other):
        """project_toImagePlane_fromOther.

        Project a point in the coordinate system of Node other onto the
        camera's image plane.

        Args:
            point_inOther (torch.Tensor): (3,) tensor corresponding to
            coordinates of a point in the coordinate system of Node other.

            other (Node): node in the pose graph in which the coordinates of
            the input point are represented.

        Returns:
            torch.Tensor: (2,) tensor corresponding to coordinates of
            projection in the coordinates of the image plane.
        """
        transform_toOther_fromCamera = \
            self.get_transform_toOther_fromSelf(other)
        point_inCamera = \
            transform_toOther_fromCamera.inverse_transform(point_inOther)

        return self.project_toImagePlane_fromCamera(point_inCamera)

    def project_ellipsoid_toImagePlane(self, ellipsoid_node):
        """project_ellipsoid_toImagePlane.

        Project and ellipsoid into its occluding contour.

        Args:
            ellipsoid_node (Ellipsoid): Ellipsoid node.
        """

        Q_inCamera = ellipsoid_node.get_ellipsoid_matrix_inOther(self)
        C_inImagePlane = \
            torch.linalg.pinv(torch.linalg.pinv(Q_inCamera)[:3, :3])

        return \
            primitives.Ellipse.create_from_homogeneous_matrix_inPlane(
                self.image_plane, C_inImagePlane
            )

    def project_ellipse_toImagePlane(self, ellipse_node):
        """project_ellipse_toImagePlane.

        Project an ellipse in 3D onto an ellipse in the camera image plane.

        Args:
            ellipse_node (Ellipse): Ellipse node.
        """

        # Overview of the procedure:
        #
        # In the coordinate system of ellipse node plane, which is assumed to
        # be the ellipse's parent node, the ellipse is represented by an
        # invertible 3x3 symmetric matrix C. C is partitioned as
        #
        #     3x2   3x1
        #      |     |
        #      V     V
        # C = [C1,   c2] <- 2x3
        #     [c2.T, c3] <- 1x3.
        #
        # We denote the inverse of C by D, which we partition as
        #
        #     3x2   3x1
        #      |     |
        #      V     V
        # D = [D1,   d2] <- 2x3
        #     [d2.T, d3] <- 1x3.
        #
        # Given an arbitrary real number s, we create an ellipsoid with 4x4
        # matrix Q in the coordinate system of the ellipse plane given by
        #
        #           4x1
        #            |
        #            V
        #     [C1,   0, c2]
        # Q = [0.T,  s,  0] <- 1x4
        #     [c2.T, 0, c3].
        #
        # Let x2_h = [X, Y, 1].T be a 2D homogeneous point represented in the
        # coordinate system of the ellipse plane node, and let X3_h = [X, Y, Z,
        # 1].T be a 3D homogeneous point the coordinate system of the ellipse
        # plane for some value of Z. For X3_h to lie on the quadric represented
        # by Q we must have
        #
        # X3_h.T @ Q @ X3_h = x2_h.T @ C @ x2_h + s * Z^2.
        #
        # For points in the vicinity of the ellipse, x2_h.T @ C @ x2_h is
        # bounded, and therefore as s grows the only points in the vicinity of
        # the ellipse that will be zeros of the above equation are those for
        # which Z approaches zero. In other words, the ellipsoid represented by
        # Q becomes "flat", and the projection its occluding contour will
        # approach the projection of the original ellipse as s grows. Let
        # T_toCamera_fromPlane be the 4x4 matrix representing the
        # transformation of from the ellipse plane to the camera coordinate
        # systems, and T_toPlane_fromCamera its inverse. Therefore, in the
        # coordinate system of the camera the matrix Q_ of this flat quadric is
        #
        # Q_ = T_toPlane_fromCamera.T @ Q @ T_toPlane_fromCamera.
        #
        # It is well known :-) that the equation of the projection C_ of the
        # occluding contour of a quadric Q_ represented in the coordinate
        # system of a camera onto the camera's image plane is given by
        #
        # C_ = ([I | 0] @ Q_^(-1) @ [I | 0].T)^(-1).
        #
        # Therefore, we have
        #
        # C_(s) = ([I | 0] @ T_toCamera_fromPlane @
        #          [D1,     0, d2]
        #          [0.T , 1/s,  0] @ T_toCamera_fromPlane.T @ [I | 0].T)^(-1)
        #          [d2.T,   0, d3]
        #
        # which, for s -> infinity, yields
        #
        # C_ = ([I | 0] @ T_toCamera_fromPlane @
        #       [D1,   0, d2]
        #       [0.T , 0,  0] @ T_toCamera_fromPlane.T @ [I | 0].T)^(-1)
        #       [d2.T, 0, d3]

        transform_toCamera_fromPlane = \
            ellipse_node.parent.get_transform_toOther_fromSelf(self)
        T_toCamera_fromPlane = transform_toCamera_fromPlane.transform_matrix

        C = ellipse_node.get_homogeneous_matrix_inPlane()
        D = torch.linalg.pinv(C)  # Pinv propagates gradients better than inv?
        D1 = D[:2, :2]
        d2 = D[:2, -1].view((2, 1))
        d3 = D[-1, -1].view((1, 1))
        Q_inv = \
            torch.vstack(
                (
                    torch.hstack((D1, torch.zeros((2, 1)), d2)),
                    torch.zeros((1, 4)),
                    torch.hstack((d2.T, torch.zeros((1, 1)), d3))
                )
            )

        C_inv = (T_toCamera_fromPlane @ Q_inv @ T_toCamera_fromPlane.T)[:3, :3]
        C_ = torch.linalg.pinv(C_inv)

        return \
            primitives.Ellipse.create_from_homogeneous_matrix_inPlane(
                self.image_plane, C_
            )

    def compute_direction_toCamera_fromImagePlane(self, point_inImage):
        """compute_direction_toCamera_fromImagePlane.

        Compute the direction of a ray from the optical center of the camera
        through a point on the image plane in the coordinate system of the
        camera.

        Args:
            point_inImage (torch.Tensor): (2,) tensor corresponding to
            normalized coordinates of point on image plane.

        Returns:
            torch.Tensor: (3,) tensor corresponding to the direction of the ray
            through the point on the image plane, in the coordinate system of
            the camera.
        """
        return torch.hstack((point_inImage, core.T1.view(1)))

    def compute_direction_toParent_fromImagePlane(self, point_inImage):
        """compute_direction_toParent_fromImagePlane.

        Compute the direction of a ray from the optical center of the camera
        through a point on the image plane, in the coordinate system of the
        camera's SubsystemModel parent node.

        Args:
            point_inImage (torch.Tensor): (2,) tensor corresponding to
            normalized coordinate of point on image plane.

        Returns:
            torch.Tensor: (3,) tensor corresponding to the direction of the ray
            through the point on the image plane, in the coordinate system of
            the camera's SubsystemModel parent node.
        """
        direction_inCamera = \
            self.compute_direction_toCamera_fromImagePlane(point_inImage)

        return \
            self.transform_toParent_fromSelf.rotation.transform(
                direction_inCamera
            )

    def get_optical_center_inOther(self, other):
        """get_optical_center_inOther.

        Get the coordinates of the camera's optical center in the coordinate
        system of node other.

        Args:
            other (Node): node in the pose graph in which to get th coordinates
            of the camera's optical center.

        Returns:
            torch.Tensor: (3,) tensor with coordinates of camera's optical
            center in the coordinate system of node other.
        """
        return self.optical_center.get_coordinates_inOther(other)

    def get_optical_center_inParent(self):
        """get_optical_center_inParent.

        Get the optical center of the camera in the coordinate system of the
        camera's parent node in the pose graph.

        Returns:
            torch.Tensor: (3,) tensor with coordinates of camera optical center
            in the coordinate system of the camera's SubsystemModel parent
            node.
        """
        return self.optical_center.get_coordinates_inOther(self.parent)

    def get_optical_axis_inOther(self, other):
        """get_optical_axis_inOther.

        Get the coordinates of the camera's optical axis (a direction) in
        the coordinate system of node other.

        Args:
            other (Node): node in which to represent the direction of the
            camera's optical axis.
        """

        optical_axis_inCamera = torch.tensor([0.0, 0.0, 1.0])

        transform_toOther_fromCamera = \
            self.get_transform_toOther_fromSelf(other)

        return \
            transform_toOther_fromCamera.rotation.transform(
                optical_axis_inCamera
            )

    def get_optical_axis_inParent(self):
        """get_optical_axis_inParent.

        Get the coordinates of the optical axis (a direction) in the
        coordinate system of the camera's parent node.
        """

        return self.get_optical_axis_inOther(self.parent)

    def compute_origin_and_direction_toParent_fromImagePlane(
        self, point_inImage
    ):
        """
        Compute the coordinates of the origin and direction of a ray
        corresponding to a point on the image plane, in the coordinate system
        of the camera's parent ET subsystem node.

        Args:
            torch.Tensor: (2,) tensor with coordinates of point on image plane.

        Returns:
            tuple of torch.Tensor: (2,) tuple with two (2,) torch.Tensors
            corresponding to the coordinates of the origin and direction of the
            ray through the point on the image plane, in the coordinate system
            of the camera's parent ET subsystem node.
        """
        direction_inParent = \
            self.compute_direction_toCamera_fromImagePlane(point_inImage)
        origin_inParent = self.get_optical_center_inParent()

        return origin_inParent, direction_inParent

    def forward_project_to_circle_from_ellipse_inImagePlane(
        self, ellipse_inImagePlane, radius
    ):
        """
        Determine the circle of given radius that projects into a given
        ellipse in the image plane of the normalized camera.

        Args:
            ellipse_inImagePlane (Ellipse): ellipse node whose parent is the
            image plane of the normalized camera.

            radius (torch.Tensor): radius of the circle in 3D that projects
            onto the given ellipse.
        """

        # Following the method and notation in
        # https://link.springer.com/content/pdf/10.1007/s007790200020.pdf

        # Get the ellipse matrix.
        C_inImagePlane = ellipse_inImagePlane.get_homogeneous_matrix_inPlane()

        # Obtain the first rotation.
        eigenvalues, R1 = torch.linalg.eigh(C_inImagePlane)
        # We need to ensure that we got a rotation.
        if torch.linalg.det(R1) < 0:
            # Note that the three eigenvalues are in ascending order.
            # Furthermore, the fist eigenvalue, eigenvalue[0] must be negative,
            # and the last eigenvalue, eigenvalue[2] must be positive. This is
            # a result of the fact that the basic homogeneous matrix has the
            # form
            #
            #     [1/a^2     0  0]
            # C = [    0 1/b^2  0].
            #     [    0     0 -1]
            #
            # This property must be preserved, and yet, the determinant of R1
            # must be changed to -1. This is how we do it:
            #
            # C = R1 @ diag(eigenvalues) @ R1.T
            #   = R1 @ diag([1, -1, 1]) @
            #          diag([1, -1, 1]) @ diag(eigenvalues) @ diag([1, -1, 1])
            #          diag([1, -1, 1]) @ R1.T
            #   = R1 @ diag([1, -1, 1]) @
            #          diag(eigenvalues) @
            #          diag([1, -1, 1]) @ R1.T
            #   = R1_ @ diag(eigenvalues) @ R1_T.

            R1 = R1 @ torch.diag(torch.tensor([1.0, -1.0, 1.0]))

        # Obtain the second rotation.
        diff_10 = eigenvalues[1] - eigenvalues[0]  # >= 0
        diff_21 = eigenvalues[2] - eigenvalues[1]  # >= 0
        sqrt_10 = torch.sqrt(diff_10)
        sqrt_21 = torch.sqrt(diff_21)
        theta_abs_deg = core.rad_to_deg(torch.atan2(sqrt_10, sqrt_21))

        # There are two possibilities for the rotation.
        R2_pos = core.rotation_around_y(theta_abs_deg)
        R2_neg = R2_pos.T  # This is core.rotation_around_y(-theta_abs_deg)

        # Composite rotation.
        RC_pos = R1 @ R2_pos
        RC_neg = R1 @ R2_neg

        # Distance from camera center to plane of circle.
        abs_prod_20 = -eigenvalues[2] * eigenvalues[0]  # > 0
        sqrt_abs_prod_20 = torch.sqrt(abs_prod_20)
        dist = torch.abs(eigenvalues[1]) / sqrt_abs_prod_20 * radius

        # Distance from foot of circle plane in camera coordinate system to
        # center of circle
        delta = sqrt_10 * sqrt_21 / sqrt_abs_prod_20 * radius

        # Coordinates of center of circle in camera coordinate system. Warning!
        # There is a typo on the paper, or at least a case that wasn't
        # discussed. Whereas the sign of dist is clear (positive, the target is
        # in front of the camera), the sign of delta must be adjusted according
        # to whether we take RC_pos or RC_neg.
        center_circle_inRotatedPlane_pos = \
            torch.hstack((delta, torch.tensor(0.0), dist))
        center_circle_inRotatedPlane_neg = \
            torch.hstack((-delta, torch.tensor(0.0), dist))
        center_circle_inCamera_pos = RC_pos @ center_circle_inRotatedPlane_pos
        center_circle_inCamera_neg = RC_neg @ center_circle_inRotatedPlane_neg

        # Normal to circle in camera coordinate system.
        normal_circle_inCamera_pos = RC_pos[:, -1]
        normal_circle_inCamera_neg = RC_neg[:, -1]

        # We should enforce that the center of the circle is in front of the
        # camera.
        if center_circle_inCamera_neg[-1] < 0:
            center_circle_inCamera_neg = -center_circle_inCamera_neg
            normal_circle_inCamera_neg = -normal_circle_inCamera_neg

        if center_circle_inCamera_pos[-1] < 0:
            center_circle_inCamera_pos = -center_circle_inCamera_pos
            normal_circle_inCamera_pos = -normal_circle_inCamera_pos

        return \
            (center_circle_inCamera_pos, normal_circle_inCamera_pos), \
            (center_circle_inCamera_neg, normal_circle_inCamera_neg)
