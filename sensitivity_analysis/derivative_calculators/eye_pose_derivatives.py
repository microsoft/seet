"""Derivative of glint, pupil, and limbus with respect to eye pose.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from sensitivity_analysis.derivative_calculators import \
    basic_derivatives
import torch


class EyePoseDerivatives(basic_derivatives.BasicDerivatives):
    """Derivative of eye measurements with respect to eye-pose parameters.

    The main goal of this class is the computation of the derivatives of glint,
    pupil, and limbus points with respect to eye-pose parameters.

    Eye-pose parameters consist of the location of the eye's rotation center
    and the two angles describing gaze direction.
    """

    @classmethod
    def get_num_parameters(cls):
        """Returns the number of params. w.r.t. which derivatives are computed.

        This is a method not a variable because it allows us to compute the
        value on the fly from data in self.derivative_data.
        """

        # There are 6 extrinsic camera parameters. 3 for the axis angle of
        # rotation, 3 for the camera position.
        # There are 5 eye-pose parameters. 2 for the gaze direction, 2 for the
        # eye position.
        return 5

    def _compute_d_x_d_parameters(self, x, x_name):
        """Computation of derivatives of glint, pupil, or limbus measurement,
        as defined by the string x_name, with respect to eye-pose parameters.

        Args:
            x (torch.Tensor): (2,) tensor corresponding to a glint, pupil, or
            limbus parameter.

            x_name (str): dummy argument to make signature of method the same
            as that of abstract method in base class.

        Returns:
            torch.tensor: (2, 5) torch tensor corresponding to the derivative
            of the measurement x with respect to the eye-pose parameters.
        """

        d_x_d_rotation_deg = \
            core.compute_auto_jacobian_from_tensors(
                x, self.eye_pose_parameters.angles_deg
            )

        d_x_d_translation_mm = \
            core.compute_auto_jacobian_from_tensors(
                x, self.eye_pose_parameters.translation
            )

        return torch.hstack((d_x_d_rotation_deg, d_x_d_translation_mm))
