"""Derivative of glints, pupil, and limbus with respect to eye shape.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from seet.sensitivity_analysis.derivative_calculators import \
    basic_derivatives, data_wrapper
import torch


class EyeShapeDerivatives(basic_derivatives.BasicDerivatives):
    """Derivatives of eye measurements with respect to eye-shape parameters.

    The main goal of this class is the computation of the derivatives of glint,
    pupil, and limbus points with respect to eye-shape parameters.

    Eye-shape parameters consist of distance parameters, curvature parameters,
    and radii parameters.
    """

    def __init__(self, derivative_data: data_wrapper.DataWrapper):
        """Create some references to shorten the names of same variables.

        Args:
            derivative_data (data_wrapper.DataWrapper): data with respect to
            which we wish to compute derivatives.
        """

        super().__init__(derivative_data)

        self.distance_from_rotation_center_to_cornea_center = \
            self.eye_shape_parameters.\
            distance_from_rotation_center_to_cornea_center
        self.distance_from_rotation_center_to_pupil_plane = \
            self.eye_shape_parameters.\
            distance_from_rotation_center_to_pupil_plane
        self.distance_from_rotation_center_to_limbus_plane = \
            self.eye_shape_parameters.\
            distance_from_rotation_center_to_limbus_plane
        self.shape_parameters = self.eye_shape_parameters.shape_parameters
        self.refractive_index = self.eye_shape_parameters.refractive_index
        self.pupil_radius = self.eye_shape_parameters.pupil_radius
        self.limbus_radius = self.eye_shape_parameters.limbus_radius

    @classmethod
    def get_num_parameters(cls):
        """Returns the number of params. w.r.t. which derivatives are computed.

        This is a method not a variable because it allows us to compute the
        value on the fly from data in self.derivative_data.
        """

        # There are 9 eye-shape parameters. 1 for the distance from the eye
        # rotation center to the cornea center, 1 for the distance from the eye
        # rotation center to the pupil plane, 1 for the distance from the eye
        # rotation center to the limbus plane, 3 for the cornea shape
        # parameters, 1 for the cornea refractive index, 1 for the pupil
        # radius, and 1 for the limbus radius.
        return 9

    def _compute_d_x_d_parameters(self, x, x_name):
        """Computation of derivative of glint, pupil, or limbus measurement, as
        defined by the string x_name, with respect to eye-shape parameters.

        Args:
            x (torch.Tensor): (2,) tensor corresponding to a glint, pupil, or
            limbus parameter.

            x_name (str): string with value 'glint', 'pupil', or 'limbus'.

        Returns:
            torch.tensor: (2, 9) torch tensor corresponding to the derivative
            of the measurement x with respect to the eye-shape parameters.
        """

        if x_name.lower() == "pupil":
            # Derivative with respect to distance from rotation center to
            # cornea apex.
            d_x_d_dist_from_rot_center_to_cornea_center = \
                core.compute_auto_jacobian_from_tensors(
                    x,
                    self.distance_from_rotation_center_to_cornea_center
                ).view((2, 1))

            # Derivative with respect to distance from rotation center to pupil
            # plane.
            d_x_d_dist_from_rot_center_to_pupil_plane = \
                core.compute_auto_jacobian_from_tensors(
                    x,
                    self.distance_from_rotation_center_to_pupil_plane
                ).view((2, 1))

            # Limbus plane does not influence pupil.
            d_x_d_dist_from_rot_center_to_limbus_plane = \
                torch.zeros(2, 1)

            # Derivative with respect to cornea shape parameters.
            d_x_d_shape_parameters = \
                core.compute_auto_jacobian_from_tensors(
                    x,
                    self.eye_shape_parameters.shape_parameters
                )

            # Derivative with respect to cornea refractive index.
            d_x_d_refractive_index = \
                core.compute_auto_jacobian_from_tensors(
                    x, self.refractive_index
                ).view((2, 1))

            # Derivative with respect to pupil radius in 3D has already been
            # computed
            d_x_d_pupil_radius = \
                core.compute_auto_jacobian_from_tensors(
                    x, self.pupil_radius
                ).view((2, 1))

            # Limbus radius does not influence pupil.
            d_x_d_limbus_radius = torch.zeros(2, 1)

        elif x_name.lower() == "limbus":
            # Derivative with respect to distance from rotation center to
            # cornea center.
            d_x_d_dist_from_rot_center_to_cornea_center = \
                core.compute_auto_jacobian_from_tensors(
                    x, self.distance_from_rotation_center_to_cornea_center
                ).view((2, 1))

            # Pupil plane does not influence limbus.
            d_x_d_dist_from_rot_center_to_pupil_plane = \
                torch.zeros(2, 1)

            # Derivative with respect to distance from rotation center to
            # limbus plane.
            d_x_d_dist_from_rot_center_to_limbus_plane = \
                core.compute_auto_jacobian_from_tensors(
                    x, self.distance_from_rotation_center_to_limbus_plane
                ).view((2, 1))

            # Shape parameters to not influence limbus
            d_x_d_shape_parameters = torch.zeros(2, 3)

            # Refractive index does not influence limbus.
            d_x_d_refractive_index = torch.zeros(2, 1)

            # Pupil radius does not influence limbus.
            d_x_d_pupil_radius = torch.zeros(2, 1)

            # Derivative with respect to limbus radius in 3D already computed.
            d_x_d_limbus_radius = \
                core.compute_auto_jacobian_from_tensors(
                    x, self.limbus_radius
                ).view((2, 1))

        else:
            d_x_d_dist_from_rot_center_to_cornea_center = \
                core.compute_auto_jacobian_from_tensors(
                    x, self.distance_from_rotation_center_to_cornea_center
                ).view((2, 1))

            # Pupil plane does not influence glints.
            d_x_d_dist_from_rot_center_to_pupil_plane = \
                torch.zeros(2, 1)

            # Limbus plane does not influence glints.
            d_x_d_dist_from_rot_center_to_limbus_plane = \
                torch.zeros(2, 1)

            d_x_d_shape_parameters = \
                core.compute_auto_jacobian_from_tensors(
                    x, self.shape_parameters
                )

            # Refractive index does not influence glints.
            d_x_d_refractive_index = torch.zeros(2, 1)

            # Pupil radius does not influence glints.
            d_x_d_pupil_radius = torch.zeros(2, 1)

            # Limbus radius does not influence glints.
            d_x_d_limbus_radius = torch.zeros(2, 1)

        # Units of the parameters with respect to which derivatives were
        # computed are
        #
        # dist_from_rot_center_to_cornea_center: mm
        # dist_from_rot_center_to_pupil_plane: mm
        # dist_from_rot_center_to_limbus_plane: mm
        # shape_parameters: mm
        # refractive_index: adimensional
        # pupil_radius: mm
        # limbus_radius: mm
        return \
            torch.hstack(
                (
                    d_x_d_dist_from_rot_center_to_cornea_center,
                    d_x_d_dist_from_rot_center_to_pupil_plane,
                    d_x_d_dist_from_rot_center_to_limbus_plane,
                    d_x_d_shape_parameters,
                    d_x_d_refractive_index,
                    d_x_d_pupil_radius,
                    d_x_d_limbus_radius
                )
            )
