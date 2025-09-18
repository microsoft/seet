"""Derivatives of measurements with respect to pupil lifting parameters.

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


from sensitivity_analysis.derivative_calculators import \
    basic_lifting_derivatives, data_wrapper


class PupilLiftingDerivatives(
    basic_lifting_derivatives.BasicLiftingDerivatives
):
    """Derivatives of eye features with respect to pupil lifting parameters.

    The pupil lifting parameters are the parameterizing angles of pupil points
    along their contours.
    """

    def __init__(self, derivative_data: data_wrapper.DataWrapper):
        """Set number of parameters w.r.t. which derivatives are computed.

        Args:
            derivative_data (data_wrapper.DataWrapper): data required for the
            computation of derivatives.
        """

        super().__init__(derivative_data)

        self.derivative_data.pupil_inPixels = []

    def get_num_parameters(self):
        """Returns the number of params. w.r.t. which derivatives are computed.

        This is a method not a variable because it allows us to compute the
        value on the fly from data in self.derivative_data.
        """

        # The number of lifting parameters is not known until the pupil is
        # computed.
        num_parameters = 0
        for limbus_point in self.derivative_data.pupil_inPixels:
            if limbus_point is not None:
                num_parameters += 1

        return num_parameters

    # Rather than overwrite the base-class method _compute_d_x_d_parameters, we
    # directly overwrite _compute_d_all_x_d_parameters, because the derivative
    # with respect to lifting parameters are extremely sparse, and we can
    # explore that.
    def _compute_d_all_x_d_parameters(self, x, x_name):
        """Computation of glint, pupil, or limbus measurements, as defined by
        the string x_name, with respect to pupil lifting parameters.

        Args:
            x (list of torch.Tensor): list of (2,) torch tensor corresponding
            to glint, pupil, or limbus measurements.

            x_name (str): one of 'glints', 'pupil', or 'limbus', specifying the
            type of parameter the derivatives of which are to be computed.

        Returns:
            torch.Tensor: list of None or (2, 1) tensors corresponding to the
            derivative of the measurement with respect to its parameterizing
            angle.
        """

        return self._compute_d_all_x_d_lifting(x, x_name, "pupil")
