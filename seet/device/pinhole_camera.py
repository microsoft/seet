"""pinhole_camera.py.

Class representing a pinhole camera.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from seet.device import device_configs
from seet.device import normalized_camera
import json
import os
import torch


class PinholeCamera(normalized_camera.NormalizedCamera):
    """PinholeCamera.

    Class for a pinhole camera. A pinhole camera consists of a normalized
    camera plus intrinsic parameters.

    The origin of the camera coordinate system is the camera optical center,
    and its orientation is naturally defined by its optical axis and the
    orientation of its image plane.

    The coordinates in the image plane have units of pixel. Pixel with
    coordinates (0, 0) corresponds to the *center* of the top-left pixel. The
    coordinates of the principal point of a (400 x 400)-pixel camera with
    principal point at the center will be (199.5, 199.5). If the focal lengths
    of the camera are 200 and 200 pixels, the field of view of the camera is 90
    degrees.
    """

    def __init__(
        self,
        subsystem_model,
        transform_toSubsystemModel_fromCamera,
        name="",
        parameter_file_name=os.path.join(
            device_configs.DEVICE_DIR,
            r"default_device/default_left_camera.json"
        ),
        requires_grad=False
    ):
        """
        Initialize a pinhole-camera object. A pinhole camera is a normalized
        camera plus intrinsic parameters. The intrinsic parameters are focal
        lengths in x and y, and the x and y coordinates of the principal point.
        Focal length and principal point have units of pixel.

        Args:
            subsystem_model (SubsystemModel): SubsystemModel object of which
            the pinhole camera is a child node.

            transform_toSubsystemModel_fromCamera (groups.SE3): Element of
            SE(3) corresponding to the transformation from the coordinate
            system of the camera to that of the eye-tracking subsystem to which
            the camera is attached.

            name (str, optional): name of object.

            parameter_file_name (str, alt): path to parameter file of pinhole
            camera. Defaults to "default_left_camera.json". It may be
            overwritten by values in parameters_dict.

            requires_grad (bool, optional): if True, gradients with respect to
            intrinsic parameters will propagated. Bypasses configuration in
            parameter_file_name. Default is None.
        """

        self.parameter_file_name = parameter_file_name

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            camera_parameters = json.load(parameter_file_stream)

        # Name the camera.
        if "name" in camera_parameters.keys():
            name = camera_parameters["name"]
        else:
            name = name

        super().__init__(
            subsystem_model, transform_toSubsystemModel_fromCamera, name=name
        )

        # Intrinsic parameters. Have to be there, let it break otherwise.
        pinhole_parameters = camera_parameters["pinhole parameters"]
        if requires_grad is None or requires_grad is False:
            if "requires grad" in pinhole_parameters.keys() and \
                    pinhole_parameters["requires grad"]:
                requires_grad = True
            else:
                requires_grad = False

        focal_lengths = \
            torch.tensor(
                pinhole_parameters["focal lengths"],
                requires_grad=requires_grad
            )
        principal_point = \
            torch.tensor(
                pinhole_parameters["principal point"],
                requires_grad=requires_grad
            )
        # This create self.focal_lengths and self.principal_point.
        self.set_pinhole_intrinsics(focal_lengths, principal_point)

        self.resolution = \
            torch.tensor(
                pinhole_parameters["resolution"],
                requires_grad=False  # Number of pixels is fixed.
            )

        self.lower_bounds = torch.tensor([-0.5, -0.5])
        self.upper_bounds = self.resolution.clone() + self.lower_bounds

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary of keyword parameters.
        """

        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}

    def set_pinhole_intrinsics(self, focal_lengths=None, principal_point=None):
        """set_pinhole_intrinsics.

        Set the focal lengths and principal point of the pinhole camera.

        Args:
            focal_lengths (torch.Tensor, optional): (2,) tensor corresponding
            to values of the x and y focal lengths of the camera in pixel
            units. Defaults to None, in which case the focal lengths are not
            changed.

            principal_point (torch.Tensor, optional): (2,) tensor corresponding
            to the coordinates of the camera principal point in pixel units.
            Defaults to None, in which case the principal point is not changed.
        """
        if focal_lengths is not None:
            self.focal_lengths = focal_lengths

        if principal_point is not None:
            self.principal_point = principal_point

    def get_point_inPixels(self, point_inImage):
        """get_point_inPixels.

        Get the pixel coordinates of a point in normalized image
        coordinates.

        Args:
            point_inImage (torch.Tensor): (2,) tensor corresponding to point in
            the normalized coordinates of the image plane.

        Returns:
            torch.Tensor: (2,) tensor corresponding to point in pixel
            coordinates.
        """
        return self.focal_lengths * point_inImage + self.principal_point

    def get_point_inImagePlane(self, point_inPixels):
        """get_point_inImagePlane.

        Get the normalized coordinates of a point in pixel coordinates.

        Args:
            point_inPixels (torch.Tensor): (2,) tensor corresponding to point
            in pixel coordinates.

        Returns:
            torch.Tensor: (2,) tensor corresponding to point in normalized
            image coordinates.
        """
        return (point_inPixels - self.principal_point) / self.focal_lengths

    def project_toPixels_fromCamera(self, point_inCamera):
        """"
        "Project a point in the coordinate system of the camera onto the
        camera's pixel coordinate system.

        Args:
            point_inCamera (torch.Tensor): (3,) tensor corresponding to
            coordinates of a point in the coordinate system of the camera.

        Returns:
            torch.Tensor: (2,) tensor corresponding to the pixel coordinates of
            the projected point.
        """
        point_inImage = self.project_toImagePlane_fromCamera(point_inCamera)

        return self.get_point_inPixels(point_inImage)

    def project_all_toPixels_fromCamera(self, all_points_inCamera):
        """Project all points in a list into pixel coordinates.

        Args:
            all_points_inCamera (list of torch.Tensor): list of (3,) torch
            tensors representing coordinates of points in the coordinate system
            of the camera, to be projected into pixel coordinates. May contain
            none values, which are projected as None.
        """

        all_points_inPixels = []
        for point_inCamera in all_points_inCamera:
            if point_inCamera is None:
                all_points_inPixels = all_points_inPixels + [None, ]
                continue

            point_inPixels = \
                self.project_toPixels_fromCamera(point_inCamera)
            all_points_inPixels = all_points_inPixels + [point_inPixels, ]

        return all_points_inPixels

    def project_toPixels_fromParent(self, point_inParent):
        """project_toPixels_fromParent.

        Project a point in the coordinate system of the camera's parent ET
        subsystem node onto the camera's pixel coordinate system.

        Args:
            point_inParent (torch.Tensor): (3,) tensor corresponding to
            coordinates of a point in the coordinate system of the camera's
            parent ET subsystem node.

        Returns:
            torch.Tensor: (2,) tensor corresponding to the pixel coordinates of
            the projected point.
        """
        point_inImage = self.project_toImagePlane_fromParent(point_inParent)

        return self.get_point_inPixels(point_inImage)

    def project_toPixels_fromOther(self, point_inOther, other):
        """project_toPixels_fromOther.

        Project a point in the coordinate system of Node other onto the
        camera's pixel coordinate system.

        Args:
            point_inOther (torch.Tensor): (3,) tensor corresponding to
            coordinates of a point in the coordinate system of Node other.

            other (Node): node in the pose graph in which the coordinates of
            the input point are represented.

        Returns:
            torch.Tensor: (2,) tensor corresponding to the pixel coordinates of
            the projected point.
        """
        point_inImage = \
            self.project_toImagePlane_fromOther(point_inOther, other)

        return self.get_point_inPixels(point_inImage)

    def project_all_toPixels_fromOther(self, all_points_inOther, other):
        """project_all_toPixels_fromOther.

        Project all 3D points in a list of points, all in the coordinate system
        of other, into a list of 2D pixel coordinates.

        Args:
            all_points_inOther (list of torch.Tensor): list of (3,) torch
            tensors representing coordinates of points in the coordinate system
            of other, to be projected into pixel coordinates. May contain none
            values, which are projected as None.

            other (Node): node in which the coordinates of the points are
            represented.
        """

        all_points_inPixels = []
        for point_inOther in all_points_inOther:
            if point_inOther is None:
                all_points_inPixels = all_points_inPixels + [None, ]
                continue

            point_inPixels = \
                self.project_toPixels_fromOther(point_inOther, other)
            all_points_inPixels = all_points_inPixels + [point_inPixels, ]

        return all_points_inPixels

    def compute_direction_toCamera_fromPixels(self, point_inPixels):
        """compute_direction_toCamera_fromPixels.

        Compute the direction of a ray from the optical center of the camera
        through a point in the camera's pixel coordinate system, in the camera
        own coordinate system.

        Args:
            point_inPixels (torch.Tensor): (2,) tensor corresponding to the
            pixel coordinates of a point in the image.

        Returns:
            torch.Tensor: (3,) tensor corresponding to the direction of the ray
            through the point in the camera's pixel coordinate system, in the
            coordinate system of the camera.
        """
        point_inImage = self.get_point_inImagePlane(point_inPixels)

        return self.compute_direction_toCamera_fromImagePlane(point_inImage)

    def compute_direction_toParent_fromPixels(self, point_inPixels):
        """compute_direction_toParent_fromPixels.

        Compute the direction of a ray from the optical center of the camera
        through a point the camera's pixel coordinate system, in the coordinate
        system of the camera's parent ET subsystem node.

        Args:
            point_inPixels (torch.Tensor): (2,) tensor corresponding to the
            pixel coordinates of a point in the image.

        Returns:
            torch.Tensor: (3,) tensor corresponding to the direction of the ray
            through the point in the camera's pixel coordinate system, in the
            coordinate system of the camera's parent ET subsystem node.
        """
        point_inImage = self.get_point_inImagePlane(point_inPixels)

        return self.compute_direction_toParent_fromImagePlane(point_inImage)

    def compute_origin_and_direction_inParent_fromPixels(self, point_inPixels):
        """compute_origin_and_direction_inParent_fromPixels.

        Compute the coordinates of the origin and direction of a ray
        corresponding to a point on the camera's pixel coordinate, in the
        coordinate system of the camera's parent ET subsystem node.

        Args:
            point_inPixels (torch.Tensor): (2,) tensor corresponding to the
            pixel coordinates of a point in the image.

        Returns:
            tuple of torch.Tensor: (2,) tuple with two (2,) tensors
            corresponding to the coordinates of the origin and direction of the
            ray through the input point, in the coordinate system of the
            camera's parent ET subsystem node.
        """
        point_inImage = self.get_point_inImagePlane(point_inPixels)

        return \
            self.compute_origin_and_direction_toParent_fromImagePlane(
                point_inImage
            )

    def compute_origin_and_direction_inOther_fromPixels(
        self, other, point_inPixels
    ):
        """Compute origin and direction of ray through pixel in other coords.

        Compute the coordinates of the origin and direction of a ray
        corresponding to a point on the camera's pixel coordinates, in the
        coordinate system of other.

        Args:
            other (Node): node in which the coordinates of the output origin
            and direction are represented.

            point_inPixels (torch.Tensor): (2,) tensor corresponding to the
            pixel coordinates of a point in the image.

        Returns:
            tuple of torch.Tensor: (2,) tuple with two (2,) tensors
            corresponding to the coordinates of the origin and direction of the
            ray through the input point, in the coordinate system of the node
            other.
        """

        origin_inParent, \
            direction_inParent = \
            self.compute_origin_and_direction_inParent_fromPixels(
                point_inPixels
            )

        transform_toOther_fromParent = \
            self.parent.get_transform_toOther_fromSelf(other)

        origin_inOther = \
            transform_toOther_fromParent.transform(origin_inParent)
        direction_inOther = \
            transform_toOther_fromParent.rotation.transform(direction_inParent)

        return origin_inOther, direction_inOther

    def is_point_inCamera_in_field_of_view(self, point_inCamera):
        """is_point_inCamera_in_field_of_view.

        Check wether a point is in the camera's field of view

        Args:
            point_inCamera (torch.Tensor): (3,) tensor corresponding to
            coordinates of a point in the camera coordinate system.

        Returns:
            bool: True if the point is in the camera's field of view, False
            otherwise.
        """
        if point_inCamera[-1] <= 0:
            return torch.tensor(False)

        point_inImage = self.project_toImagePlane_fromCamera(point_inCamera)
        point_inPixels = self.get_point_inPixels(point_inImage)

        return \
            torch.all(
                torch.gt(point_inPixels, self.lower_bounds)
            ) and \
            torch.all(
                torch.lt(point_inPixels, self.upper_bounds)
            )

    def is_point_inParent_in_field_of_view(self, point_inParent):
        """is_point_inParent_in_field_of_view.

        Check whether a point in the coordinate system of the camera's
        parent ET subsystem is within the frustum of the camera's field of
        view.

        Args:
            point_inParent (torch.Tensor): (3,) tensor corresponding to
            coordinates of point in the coordinate system of the camera's
            parent ET subsystem.

        Returns:
            bool: True if the point is in the camera's field of view, False
            otherwise.
        """
        point_inCamera = \
            self.transform_toParent_fromSelf.inverse_transform(point_inParent)

        return self.is_point_inCamera_in_field_of_view(point_inCamera)
