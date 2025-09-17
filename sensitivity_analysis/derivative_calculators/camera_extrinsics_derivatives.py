"""Derivative of glints, pupil, and limbus with respect to extrinsic params.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from sensitivity_analysis.derivative_calculators import \
    basic_derivatives, data_wrapper
import torch


class CameraExtrinsicsDerivatives(basic_derivatives.BasicDerivatives):
    """Derivatives of eye features with respect to extrinsic camera parameters.

    The main goal of this class is the computation of the derivatives of glint,
    pupil, and limbus points with respect to extrinsic camera parameters.
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

        # There are 6 extrinsic camera parameters. 3 for the axis angle of
        # rotation, 3 for the camera position.
        return 6

    def _compute_d_x_d_parameters(self, x, x_name):
        """Compute derivatives of x with respect to extrinsic parameters.

        Args:
            x (torch.Tensor): (2,) tensor corresponding to a glint, pupil, or
            limbus parameter.

            x_name (str): string with value 'glint', 'pupil', or 'limbus'.

        Returns:
            torch.tensor: (2, 6) torch tensor corresponding to the derivative
            of the measurement x with respect to the camera orientation and
            location.
        """

        d_x_d_axis_angle_rad = \
            core.compute_auto_jacobian_from_tensors(
                x, self.camera_extrinsics.axis_angle
            )
        # d_x/d_(angle_rad) = d_x/d_(angle_mrad / 1000)
        #                   = 1000 * d_x/d_(angle_mrad)
        d_x_d_axis_angle_mrad = d_x_d_axis_angle_rad / 1000

        d_x_d_translation_mm = \
            core.compute_auto_jacobian_from_tensors(
                x, self.camera_extrinsics.translation
            )

        d_x_d_parameters = \
            torch.hstack((d_x_d_axis_angle_mrad, d_x_d_translation_mm))

        return d_x_d_parameters
