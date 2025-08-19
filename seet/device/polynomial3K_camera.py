"""polynomial3K_camera.py.

Class representing a polynomial 3K camera.

The polynomial 3K camera is a pinhole camera with added radial distortion
parameters. These parameters consist of a distortion center and three radial
distorion coefficients. 3D points produce 2D pixels by first undergoing a
projection to normalized coordinate, then radial distortion is applied, and
finally the pinhole intrinsics are applied.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from seet.device import device_configs
from seet.device import pinhole_camera
import json
import os
import torch


class Polynomial3KCamera(pinhole_camera.PinholeCamera):
    """Polynomial3KCamera.

    Class for polynomial 3K camera.

    3D points are projected, distorted and converted to pixel coordinates. The
    distortion model takes normalized, undistorted coordinates (u, v) and maps
    them to distorted coordinate (u_, v_) through the equations:

    (du, dv) = (u, v) - (cu, cv)

    r = ||(du, dv)||

    d = 1.0 + k0*r^2, + k1*r^4 + k2*r^6

    (u_, v_) = (cu, cv) + d * (du, dv)
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
        requires_grad=None
    ):
        """
        Initialize a polynomial-3K-camera object. A polynomial 3K camera is
        a pinhole camera plus radial distortion parameters. The radial
        distortion parameters are a distoriton center in normalized image
        coordinates and radial distortion coefficients.

        Args:
            subsystem_model (SubsystemModel): SubsystemModel object of which
            the polynomial 3K camera is a child node.

            transform_toSubsystemModel_fromCamera (groups.SE3): Element of
            SE(3) corresponding to the transformation from the coordinate
            system of the camera to that of the eye-tracking subsystem to which
            the camera is attached.

            name (str, optional): name of object.

            parameter_file_name (str, optional): path to parameter file of
            polynomial 3K camera. Defaults to "default_left_camera.json".

            requires_grad (bool, optional): if true, the distortion parametes
            are assumed to be differentiable. Defaults to False.
        """
        # Initialize radial distortion parameters.
        with core.Node.open(parameter_file_name) as parameter_file_stream:
            camera_parameters = json.load(parameter_file_stream)

        # Name the camera.
        if "name" in camera_parameters.keys():
            name = camera_parameters["name"]
        else:
            name = name

        # Initialize parameters of pinhole.
        super().__init__(
            subsystem_model,
            transform_toSubsystemModel_fromCamera,
            name=name,
            parameter_file_name=parameter_file_name,
            requires_grad=requires_grad
        )

        # Distortion parameters.
        distortion_parameters = camera_parameters["distortion parameters"]

        if requires_grad is None:
            if "requires grad" in distortion_parameters.keys() and \
                    distortion_parameters["requires grad"]:
                requires_grad = True
            else:
                requires_grad = False

        distortion_center = \
            torch.tensor(
                distortion_parameters["distortion center"],
                requires_grad=requires_grad
            )
        distortion_coefficients = \
            torch.tensor(
                distortion_parameters["distortion coefficients"],
                requires_grad=requires_grad
            )
        self.set_distortion_parameters(
            distortion_center, distortion_coefficients
        )

    def set_distortion_parameters(
        self, distortion_center=None, distortion_coefficients=None
    ):
        """
        Set the distortion center and the distortion coefficients of the
        polynomial 3K camera.

        Args:
            distortion_center (torch.Tensor, optional): (2,) tensor
            corresponding to the coordinates of the distortion center in
            normalized image coordinates. Defaults to None, in which case the
            distortion center is not changed.

            distortion_coefficients (torch.Tensor, optional): (3,) tensor
            corresponding to the coordinates of the distortion coefficients of
            normalized, decentered points in image coordinates. Defaults to
            None, in which case the distortion coefficients are not changed.
        """
        if distortion_center is not None:
            self.distortion_center = distortion_center

        if distortion_coefficients is not None:
            self.distortion_coefficients = distortion_coefficients

    @staticmethod
    def compute_radial_distortion(
        decentered_point_inUndistortedImage, distortion_coefficients
    ):
        """
        Static method for computation of term d in
        x_distorted - c = d*(x_undistorted - c)

        Args:
            decentered_point_inUndistortedImage (torch.Tensor): (2,) tensor
            with normalized image coordinates of undistorted point after
            decentration, i.e., subtraction by the distortion center.

            distortion_coefficients (torch.Tensor): (3,) tensor with distortion
            coefficients.

        Returns:
            torch.float: factor d such that d * (x_undistorted -
            distortion_center) = x_distorted - distortion_center.
        """
        dist_square = torch.square(decentered_point_inUndistortedImage).sum()
        dist_fourth = dist_square * dist_square
        dist_sixth = dist_square * dist_fourth
        dist_powers = torch.stack((dist_square, dist_fourth, dist_sixth))

        return (distortion_coefficients * dist_powers).sum() + core.T1

    def distort_point_inUndistortedImage(self, point_inUndistortedImage):
        """distort_point_inUndistortedImage.

        Apply distortion parameters to obtain point in normalized distorted
        image coordinates.

        Args:
            point_inUndistortedImage (torch.Tensor): (2,) tensor in normalized
            undistorted image coordinates.

        Returns:
            torch.Tensor: (2,) tensor in normalized distorted image
            coordinates.
        """
        decentered_point_inUndistortedImage = \
            point_inUndistortedImage - self.distortion_center
        radial_distortion = \
            Polynomial3KCamera.compute_radial_distortion(
                decentered_point_inUndistortedImage,
                self.distortion_coefficients
            )

        return \
            self.distortion_center + \
            radial_distortion * decentered_point_inUndistortedImage

    class _undistort_decentered_pt_inDistortedImage(torch.autograd.Function):
        """_undistort_decentered_pt_inDistortedImage.

        Because the undistortion is the output of a fixed-point iteration,
        we need to define custom autograd functions, otherwise propagation of
        derivatives would be incorrect.
        """
        @staticmethod
        def forward(
            ctx,
            decentered_point_inDistortedImage,
            distortion_coefficients,  # Dummy; we use the camera.
            camera,
            tol,
            max_num_iter,
            create_graph
        ):
            # No computation of gradient!
            with torch.no_grad():
                # Initialization.
                decentered_point_inUndistortedImage = \
                    decentered_point_inDistortedImage.clone()

                # Fixed-point iteration.
                for _ in range(max_num_iter):
                    radial_distortion = \
                        Polynomial3KCamera.compute_radial_distortion(
                            decentered_point_inUndistortedImage,
                            camera.distortion_coefficients
                        )

                    new_decentered_point_inUndistortedImage = \
                        decentered_point_inDistortedImage - \
                        (radial_distortion - 1) * \
                        decentered_point_inUndistortedImage

                    error = torch.linalg.norm(
                        new_decentered_point_inUndistortedImage -
                        decentered_point_inUndistortedImage
                    )

                    decentered_point_inUndistortedImage = \
                        new_decentered_point_inUndistortedImage

                    if error < tol:
                        break

                ctx.save_for_backward(
                    decentered_point_inUndistortedImage,
                    distortion_coefficients
                )

                ctx.camera = camera

                ctx.create_graph = create_graph

            return decentered_point_inUndistortedImage

        @staticmethod
        def backward(ctx, grad_output):
            """backward.

            Should return:

            grad_output @ d_decentered_undist_pt_d_decentered_dist_pt
            grad_output @ d_decentered_undist_pt_d_dist_coeff
            None # No output with respect to camera
            None # No output with respect to tol
            None # No output with respect to number of iterations
            None # No output with respect to boolean flag.

            Args:
                ctx (context): context with saved data. grad_output
                (torch.Tensor): gradient of output for backprop.
            """

            # To compute d_decentered_undist_pt_d_decentered_dist_pt we compute
            # its inverse and invert it.
            decentered_point_inUndistortedImage, _ = ctx.saved_tensors
            camera = ctx.camera
            create_graph = ctx.create_graph

            # Forward computation.
            fresh_decentered_point_inUndistortedImage = \
                decentered_point_inUndistortedImage.clone().detach()
            fresh_decentered_point_inUndistortedImage.requires_grad = True
            with torch.enable_grad():
                decentered_point_inDistortedImage = \
                    camera.distort_point_inUndistortedImage(
                        fresh_decentered_point_inUndistortedImage +
                        camera.distortion_center
                    )

            d_decentered_dist_pt_d_decentered_undist_pt = \
                core.compute_auto_jacobian_from_tensors(
                    decentered_point_inDistortedImage,
                    fresh_decentered_point_inUndistortedImage,
                    create_graph=create_graph  # In case we want Hessians.
                )

            d_decentered_dist_pt_d_dist_coefficients = \
                core.compute_auto_jacobian_from_tensors(
                    decentered_point_inDistortedImage,
                    camera.distortion_coefficients,
                    create_graph=create_graph  # In case we want Hessians.
                )

            d_decentered_undist_pt_d_decentered_dist_pt = \
                torch.linalg.pinv(
                    d_decentered_dist_pt_d_decentered_undist_pt
                )

            d_decentered_undist_pt_d_dist_coefficients = \
                d_decentered_undist_pt_d_decentered_dist_pt @ \
                d_decentered_dist_pt_d_dist_coefficients

            return \
                grad_output @ d_decentered_undist_pt_d_decentered_dist_pt, \
                grad_output @ d_decentered_undist_pt_d_dist_coefficients, \
                None, \
                None, \
                None, \
                None

    def undistort_point_inDistortedImage(
        self,
        point_inDistortedImage,
        tol=core.TEPS,
        max_num_iter=30,
        create_graph=False
    ):
        """
        Remove effect of distortion to obtain point in undistorted
        normalized image coordinates.

        Args:
            point_inDistortedImage (torch.Tensor): (2,) tensor corresponding to
            point in normalized distorted image coordinates.

        Returns:
            torch.Tensor: (2,) tensor in normalized undistorted image
            coordinates.
        """
        decentered_point_inDistortedImage = \
            point_inDistortedImage - self.distortion_center

        decentered_point_inUndistortedImage = \
            self._undistort_decentered_pt_inDistortedImage.apply(
                decentered_point_inDistortedImage,
                self.distortion_coefficients,
                self,
                tol,
                max_num_iter,
                create_graph
            )

        return decentered_point_inUndistortedImage + self.distortion_center

    # Let's keep this code here for now, but preliminary investigations suggest
    # that it is 3x slower than the non-alt version.
    def alt_undistort_point_inDistortedImage(
        self, point_inDistortedImage, tol=core.TEPS * 10
    ):
        decentered_point_inDistortedImage = \
            point_inDistortedImage - self.distortion_center

        norm_ = torch.linalg.norm(decentered_point_inDistortedImage)
        if norm_ < tol:
            return point_inDistortedImage

        # We are solving |(x - c)(1 + k0*r^2 + k1*r^4 + k2*r^6)| = |(x_ - c)|.
        # Once we take the signs in to account, that will result in roots, xi
        # and -xi. Only the positive roots are plausible.
        p = \
            torch.hstack(
                (
                    -norm_,
                    core.T1,
                    core.T0,
                    self.distortion_coefficients[0],
                    core.T0,
                    self.distortion_coefficients[1],
                    core.T0,
                    self.distortion_coefficients[2]
                )
            )
        # Rather than solving the polynomial equation twice to get roots xi and
        # -xi, we solve it once and make all roots positive.
        norm = torch.abs(core.poly_solver(p))

        # Take the one that results in the least amount of distortion.
        index_min = torch.argmin(torch.abs(norm - norm_.view(1)))
        norm = norm[index_min]

        decentered_point_inUndistortedImage = \
            decentered_point_inDistortedImage / norm_ * norm

        return decentered_point_inUndistortedImage + self.distortion_center

    # By overloading the base classe's method get_point_inImagePlane and
    # get_point_inPixels we can reutilize many other base classe's methods.
    def get_point_inImagePlane(self, point_inPixels, create_graph=False):
        """get_point_inImagePlane.

        Get points in *undistorted* image plane.

        Args:
            point_inPixels (torch.Tensor): (2,) tensor corresponding to point
            in pixel coordinates.

            create_graph (bool, optional): if True, backpropagation graph of
            derivatives is compute, for computation of second derivatives.
            Default is false.

        Returns:
            torch.Tensor: (2,) tensor corresponding to a point in normalized
            image coordinates.
        """
        # First, remove effect of intrinsics.
        distorted_point_inImage = \
            super().get_point_inImagePlane(point_inPixels)

        # Then, remove effect of radial distortion.
        return \
            self.undistort_point_inDistortedImage(
                distorted_point_inImage, create_graph=create_graph
            )

    def get_point_inPixels(self, point_inImage):
        """get_point_inPixels.

        Get points in pixel coordinates from *undistorted* points in the
        image plane.

        Args:
            point_inImage (torch.Tensor): (2,) tensor corresponding to
            undistorted coordinates in the image plane.
        """
        # First, apply distortion.
        distorted_point_inImage = \
            self.distort_point_inUndistortedImage(point_inImage)

        # Then, add effect of intrinsics.
        return super().get_point_inPixels(distorted_point_inImage)
