"""Derivative of glints, pupil, and limbus with respect to LED locations.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from sensitivity_analysis.derivative_calculators import \
    basic_derivatives, data_wrapper
import torch


class LEDLocationsDerivatives(basic_derivatives.BasicDerivatives):
    """Derivatives of eye features with respect to LED locations.

    We know a priori that the derivatives of pupil and limbus points with
    respect to LED locations is zero. Therefore, the main computation is the
    derivative of glints with respect to LED coordinates. Those coordinates are
    provided in the the coordinate system of the corresponding eye-tracking
    subsystem.

    This class also provides a method to compute the derivative of the
    measurements with respect to rotations or translations of the LEDs in the
    camera coordinate system.

    For N LEDs there will be 3N parameters corresponding to the 3D locations of
    the N LEDs.
    """

    def __init__(self, derivative_data: data_wrapper.DataWrapper):
        """Create some references to shorten the names of same variables.

        Args:
            derivative_data (data_wrapper.DataWrapper): data with respect to
            which we wish to compute derivatives.
        """

        super().__init__(derivative_data)

        self.all_translations = self.led_locations.all_translations
        self.all_coordinates_inSubsystem = \
            self.led_locations.all_coordinates_inSubsystem

        self.joint_axis_angle = self.led_locations.joint_axis_angle
        self.joint_translation = self.led_locations.joint_translation
        self.leds = self.led_locations.leds

        # Transformation is an element of SE(3).
        self.num_transformation_parameters = 6

    def get_num_parameters(self):
        """Returns the number of params. w.r.t. which derivatives are computed.

        This is a method not a variable because it allows us to compute the
        value on the fly from data in self.derivative_data.
        """

        # Each LED is described by three coordinates.
        return self.leds.num * 3

    # Rather than overwrite the base-class method _compute_d_x_d_parameters, we
    # directly overwrite _compute_d_all_x_d_parameters, because the derivatives
    # with respect to LEDs are extremely sparse, and we can explore that.
    def _compute_d_all_x_d_parameters(self, x, x_name):
        """Computation of the derivative of glint, pupil, or limbus
        measurements, as defined by the string x_name, with respect to LED
        coordinates.

        Note that we set the derivative with respect to an occluded LED to
        zero. This is different from what we do for derivatives with respect to
        lifting parameters. In the latter case we set the derivative to None.
        The reason for the difference is that if an LED is not visible in a
        given image, it may be visible in another. By setting the derivatives
        with respect to occluded LEDs to zero we do not need to keep track of
        that.

        Args:
            x (list of torch.Tensor): list of (2,) torch tensors corresponding
            to glint, pupil or limbus measurements.

            x_name (str): one of 'glints', 'pupil', or 'limbus', specifying the
            type of parameter with respect the derivatives of which are to be
            computed.

        Returns:
            torch.Tensor: list of None or (2, 3*N) tensors corresponding to the
            derivative of the i-th measurement.
        """

        # Derivatives with respect to pupil or limbus points are zero.
        num_parameters = self.get_num_parameters()
        if x_name.lower() != "glints":
            return [torch.zeros(2, num_parameters) for _ in x]

        d_x_d_parameters = []
        for i, single_glint in enumerate(x):
            if single_glint is None:
                # Skip non-visible glints.
                continue

            # The structure of the derivative is
            #
            #                    i (2x3) blocks     led i  (N-i-1) (2x3) blocks
            #                            |            |            |
            #                            V            V            V
            #                    [0 0 0 ... 0 0 0   x x x   0 0 0 ... 0 0 0]
            # d_glint_i_d_leds = [0 0 0 ... 0 0 0   x x x   0 0 0 ... 0 0 0]
            #                       ^         ^       ^       ^         ^
            #                       |         |       |       |         |
            #               block   0   ...  i-1      i      i+1  ...  N-1

            left_padding = torch.zeros(2, 3 * i)
            right_padding = torch.zeros(2, num_parameters - 3 * (i + 1))

            # We save computation by computing the derivative of a single glint
            # with respect to a single LED.
            middle_value = \
                core.compute_auto_jacobian_from_tensors(
                    single_glint, self.all_coordinates_inSubsystem[i]
                )

            d_single_glint_d_led = \
                torch.hstack((left_padding, middle_value, right_padding))

            d_x_d_parameters = d_x_d_parameters + [d_single_glint_d_led, ]

        return d_x_d_parameters

    def compute_d_glints_d_leds_transformation(self):
        """Return the derivatives of glints with respect an LED transformation.

        Returns:
            list of Union(torch.Tensor | None): list of (2, 6) tensors or None,
            where None corresponds to a glint that is not visible.
        """

        d_glints_d_leds_transformation = []
        for single_glint_inPixels in self.derivative_data.all_glints_inPixels:
            if single_glint_inPixels is None:
                # Some glints are not visible
                continue

            d_single_glint_inPixels_d_leds_rotation = \
                core.compute_auto_jacobian_from_tensors(
                    single_glint_inPixels, self.joint_axis_angle
                )

            d_single_glint_inPixels_d_leds_translation = \
                core.compute_auto_jacobian_from_tensors(
                    single_glint_inPixels, self.joint_translation
                )

            d_glints_d_leds_transformation = \
                d_glints_d_leds_transformation + \
                [
                    torch.hstack(
                        (
                            d_single_glint_inPixels_d_leds_rotation,
                            d_single_glint_inPixels_d_leds_translation
                        )
                    ),
                ]

        return d_glints_d_leds_transformation

    def _compute_d_x_d_leds_transformation(self, x):
        """Derivatives of pupil or limbus with respect to LED transformation.

        Args:
            x (list of Union(None | torch.Tensor)): list with None if the
            feature point is not visible, or a (2,) torch tensor corresponding
            to the coordinates of the feature in pixels.

        Returns:
            list of torch.Tensor: list of (2, 6) zero torch tensors
            tensor.
        """

        zero = torch.zeros(2, self.num_transformation_parameters)
        d_x_d_leds_transformation = \
            [zero for single_x in x if single_x is not None]

        return d_x_d_leds_transformation

    def compute_d_pupil_d_leds_transformation(self):
        """Returns the derivatives of pupil with respect to LED transformation.

        The result is a list of either None, if the pupil point is not visible,
        or (2, 6) zero torch tensors, since the LEDs to not influence the pupil
        points.

        Returns:
            list of Union(None | torch.Tensor): list with None if the
            corresponding pupil point is not visible, or a (2, 6) zero torch
            tensor.
        """

        return \
            self._compute_d_x_d_leds_transformation(
                self.derivative_data.pupil_inPixels
            )

    def compute_d_limbus_d_leds_transformation(self):
        """Returns the derivative of limbus with respect to LED transformation.

        The result is a list of either None, if the limbus point is not
        visible, or (2, 6) zero torch tensors, since the LEDs to not influence
        the limbus points.

        Returns:
            list of Union(None | torch.Tensor): list with None if the
            corresponding limbus point is not visible, or a (2, 6) zero torch
            tensor.
        """

        return \
            self._compute_d_x_d_leds_transformation(
                self.derivative_data.limbus_inPixels
            )
