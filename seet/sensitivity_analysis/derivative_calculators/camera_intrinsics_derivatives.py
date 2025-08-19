"""Derivative of glints, pupil, and limbus with respect to intrinsic params.

The base class implements the computation of the derivative of glints, pupil,
or limbus with respect to some generic parameter.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from seet.sensitivity_analysis.derivative_calculators import \
    basic_derivatives, data_wrapper
import torch


class CameraIntrinsicsDerivatives(basic_derivatives.BasicDerivatives):
    """Derivatives of eye features with respect to intrinsic camera parameters.

    The main goal of this class is the computation of the derivatives of glint,
    pupil, and limbus points with respect to intrinsic camera parameters.

    Intrinsic camera parameters consist of pinhole parameters and radial
    distortion parameters.
    """

    def __init__(self, derivatives_data: data_wrapper.DataWrapper):
        """Set number of parameters w.r.t which derivatives are computed.

        Args:
            derivative_data (data_wrapper.DataWrapper): data required for the
            computation of derivatives.
        """

        super().__init__(derivatives_data)

    @classmethod
    def get_num_parameters(cls):
        """Returns the number of params. w.r.t. which derivatives are computed.

        This is a method not a variable because it allows us to compute the
        value on the fly from data in self.derivative_data.
        """

        # There are 9 intrinsic camera parameters. 2 for the horizontal and
        # vertical focal lengths, 2 for the principal point, 2 for the
        # distortion center, and 3 for the distortion parameters.
        return 9

    def _compute_d_x_d_parameters(self, x, x_name):
        """Compute derivatives of x with respect to intrinsic parameters.

        Args:
            x (torch.Tensors): (2,) torch tensors corresponding to a glint,
            pupil or limbus point.

            x_name (str): string with value 'glint', 'pupil', or 'limbus'.

        Returns:
            torch.tensor: (2, 9) or (1, 9) torch tensor corresponding to the
            derivative of the measurement x with respect to the camera focal
            lengths (two columns), principal point (two columns), distortion
            center (two columns), and distortion coefficients (three columns).
            The return value will be (2, 9) if the measurement is a glint, and
            (1, 9) if the measurement is a pupil or limbus point.
        """

        d_x_d_focal_lengths_pix = \
            core.compute_auto_jacobian_from_tensors(
                x, self.camera_intrinsics.focal_lengths)

        d_x_d_principal_point_pix = \
            core.compute_auto_jacobian_from_tensors(
                x, self.camera_intrinsics.principal_point
            )

        d_x_d_distortion_center_adimensional = \
            core.compute_auto_jacobian_from_tensors(
                x, self.camera_intrinsics.distortion_center
            )

        d_x_d_distortion_coefficients_adimensional = \
            core.compute_auto_jacobian_from_tensors(
                x, self.camera_intrinsics.distortion_coefficients
            )

        return \
            torch.hstack(
                (
                    d_x_d_focal_lengths_pix,
                    d_x_d_principal_point_pix,
                    d_x_d_distortion_center_adimensional,
                    d_x_d_distortion_coefficients_adimensional
                )
            )
