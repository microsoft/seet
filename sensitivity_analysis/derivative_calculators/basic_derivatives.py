"""Base class for derivative of eye measurements.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsof.com)"


from sensitivity_analysis.derivative_calculators import data_wrapper


class BasicDerivatives():
    """Base class for derivatives of glint, pupil, and limbus points.
    """

    def __init__(self, derivative_data: data_wrapper.DataWrapper):
        """Initialize common parameters.

        Args:
            derivative_data (data_wrapper.DataWrapper): data required for the
            computation of derivatives.
        """

        self.derivative_data = derivative_data

        # Easy access to some data.
        self.camera_extrinsics = self.derivative_data.camera_extrinsics
        self.camera_intrinsics = self.derivative_data.camera_intrinsics
        self.led_locations = self.derivative_data.led_locations
        self.eye_pose_parameters = self.derivative_data.eye_pose_parameters
        self.eye_shape_parameters = self.derivative_data.eye_shape_parameters

        # These parameters are made available only when features are computed.
        self.pupil_angles_rad = None
        self.limbus_angles_rad = None

    @classmethod
    def get_num_parameters(cls):
        """Returns the number of params. w.r.t. which derivatives are computed.

        This is a method not a variable because it allows us to compute the
        value on the fly from data in self.derivative_data.
        """

        raise NotImplementedError("Must be implemented by derived class.")

    def _compute_d_x_d_parameters(self, x, x_name):
        """Abstract method to be implemented by derived class.

        Args:
            x (torch.Tensor): (2,) torch tensor corresponding to a glint, pupil
            or limbus measurement.

            x_name (str): string with value 'glint', 'pupil', or 'limbus'.
        """

        raise NotImplementedError("Must be implemented by derived class.")

    def _compute_d_all_x_d_parameters(self, x, x_name):
        """Parametric method for computing derivatives of all measurements.

        Args:
            x (list of torch.Tensor): list of (2,) torch tensors corresponding
            to either pupil or limbus measurements.

            x_name (str): string with value 'glint', 'pupil', or 'limbus'.

        Returns:
            list of torch.Tensor: list of (1, 9) torch tensor, with each
            element of the list corresponding to the derivative of a one
            dimensional pupil or limbus point in pixels with respect to the
            nine intrinsic camera parameters.
        """

        d_x_d_parameters = []
        for single_x in x:
            if single_x is None:
                continue

            d_single_x_d_parameters = \
                self._compute_d_x_d_parameters(single_x, x_name)

            d_x_d_parameters = d_x_d_parameters + [d_single_x_d_parameters, ]

        return d_x_d_parameters

    def compute_d_glints_d_parameters(self):
        """Compute derivatives of glints with respect to parameters.

        The base method generate_glints_inPixels() must be run before this
        method, so that self.all_glints_inPixels is populated.

        Returns:
            list of torch.Tensor: list of (2, N) torch tensors, with each
            element in the list corresponding to the derivative of a two
            dimensional glint in pixels with respect to the N camera
            parameters. Since each glint corresponds to an LED, derivatives of
            glints that are not visible (fall outside the cornea, occluded,
            etc.) are marked in the list as None.
        """

        self.derivative_glints = \
            self._compute_d_all_x_d_parameters(
                self.derivative_data.all_glints_inPixels, "glints"
            )

        return self.derivative_glints

    def compute_d_pupil_d_parameters(self):
        """Compute derivatives of pupil with respect to some parameters.

        The base method generate_pupil_inCornea() must be run before this
        method, sot that self.pupil_inPixels is populated. The base method
        compute_d_pupil_d_pupil_radius() must also be run so that
        self.direction_d_pupil_d_pupil_radius is also populated.

        Returns:
            list of torch.Tensor: list of (1, 9) torch tensor, with each
            element of the list corresponding to the derivative of a one
            dimensional pupil point in pixels with respect to the nine
            intrinsic camera parameters. In principle, 30 points are sampled
            along the pupil. However, some pupil points may not be visible to
            to occlusion, and are marked in the list as None.
        """

        self.derivative_pupil = \
            self._compute_d_all_x_d_parameters(
                self.derivative_data.pupil_inPixels, "pupil"
            )

        return self.derivative_pupil

    def compute_d_limbus_d_parameters(self):
        """Compute derivatives of limbus with respect to intrinsic parameters.

        The base method generate_limbus_inCornea() must be run before this
        method, sot that self.limbus_inPixels is populated. The base method
        compute_d_limbus_d_limbus_radius() must also be run so that
        self.direction_d_limbus_d_limbus_radius is also populated.

        Returns:
            list of torch.Tensor: list of (1, 9) torch tensor, with each
            element of the list corresponding to the derivative of a one
            dimensional limbus point in pixels with respect to the nine
            intrinsic camera parameters.  In principle, 30 points are sampled
            along the limbus. However, some limbus points may not be visible to
            to occlusion, and are marked in the list as None.
        """

        self.derivative_limbus = \
            self._compute_d_all_x_d_parameters(
                self.derivative_data.limbus_inPixels, "limbus"
            )

        return self.derivative_limbus
