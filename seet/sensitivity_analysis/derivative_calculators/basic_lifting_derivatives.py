"""Base class for derivatives of measurements w.r.t. lifting parameters.

Limbus and pupil points are, effectively, one dimensional, in that they 'slide'
along contours. Parameterizing the contours in polar coordinates, the points
are then associated to the angle along the contour to which their correspond.
It greatly simplifies sensitivity analyses if the derivatives of these features
with respect to their parameterizing angles is available, a technique know as
lifting. For details of the lifting technique, see
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/SIGGRAPH2016-SmoothHandTracking.pdf
and references therein.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from seet.sensitivity_analysis.derivative_calculators import \
    basic_derivatives
import torch


class BasicLiftingDerivatives(basic_derivatives.BasicDerivatives):
    """Derivatives of eye features with respect to pupil lifting parameters.

    This class computes the derivatives of measurements with respect to lifting
    parameters of either pupil or limbus points.
    """

    # Rather than overwrite the base-class method _compute_d_x_d_parameters, we
    # directly overwrite _compute_d_all_x_d_parameters, because the derivative
    # with respect to lifting parameters are extremely sparse, and we can
    # explore that.
    def _compute_d_all_x_d_lifting(self, x, x_name, lifting_name):
        """Computation of glint, pupil, or limbus measurements, as defined by
        the string x_name, with respect to pupil lifting parameters.

        Note that the number of lifting parameters is variable, as it depends
        on the number of visible pupil and limbus points.

        Args:
            x (list of torch.Tensor): list of (2,) torch tensor corresponding
            to glint, pupil, or limbus measurements.

            x_name (str): one of 'glints', 'pupil', or 'limbus', specifying the
            type of parameter the derivatives of which are to be computed.

            lifting_name (str): one of 'pupil' or 'limbus', specifying the type
            of parameters with respect to which derivatives are to be computed.

        Returns:
            torch.Tensor: list of None or (2, 1) tensors corresponding to the
            derivative of the measurement with respect to its parameterizing
            angle.
        """

        lifting_name = lifting_name.lower()
        if lifting_name.lower() == "pupil":
            angles = self.derivative_data.pupil_angles_rad
        else:
            angles = self.derivative_data.limbus_angles_rad

        num_parameters = self.get_num_parameters()

        x_name = x_name.lower()
        if x_name == "glints" or x_name != lifting_name:
            # Derivatives of glints with respect to lifting parameters are
            # always zero. Also derivative of pupil (resp., limbus) points with
            # respect to limbus (resp., pupil) lifting parameters are also
            # zero.
            return \
                [
                    torch.zeros(2, num_parameters) for single_x in x
                    if single_x is not None
                ]

        d_x_d_parameters = []
        j = 0
        for i, xi in enumerate(x):
            if xi is None:
                # Derivatives with respect to an occluded feature is absent.
                continue

            # The structure of the derivative is
            #
            #            angle -> 0 1     i     N
            # d_x_d_angles =
            #        point |  0  [x 0 ... 0 ... 0]
            #              V  0  [x 0 ... 0 ... 0]
            #                 1  [0 x ... 0 ... 0]
            #                 1  [0 x ... 0 ... 0]
            #                    [...............]
            #                 i  [0 0 ... x ... 0]
            #                 i  [0 0 ... x ... 0]
            #                    [...............]
            #                 N  [0 0 ... 0 ... x]
            #                 N  [0 0 ... 0 ... x]
            #
            # The derivative of x point i with respect to angles is
            #
            #                    i zeros     i   N-i-1 zeros
            # d_x_i_d_angles = [0 0  ...  0  x  0  ...   0 0]
            #                  [0 0  ...  0  x  0  ...   0 0]

            # Stop linting complaints...
            assert (angles is not None), "Angles not initialized"
            angle_i = angles[i]

            left_padding = torch.zeros(2, j)
            right_padding = torch.zeros(2, num_parameters - j - 1)
            j += 1

            d_xi_d_angle_i = \
                torch.hstack(
                    (
                        left_padding,
                        core.compute_auto_jacobian_from_tensors(xi, angle_i),
                        right_padding
                    )
                )

            d_x_d_parameters = d_x_d_parameters + [d_xi_d_angle_i, ]

        return d_x_d_parameters
