"""Class and functions for computing covariance matrix of eye-shape parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import sensitivity_analysis.derivative_calculators as derivative_calc
import sensitivity_analysis.eye_pose_covariance as eye_pose_covariance
import sensitivity_analysis.sensitivity_analysis_configs as \
    sensitivity_analysis_configs
import os
import torch


class EyeShapeCovariance(eye_pose_covariance.EyePoseCovariance):
    """Class and for computing covariance matrix of eye-shape parameters.

    It may be strange that EyeShapeCovariance inherits from EyePoseCovariance,
    but the estimator of EyeShapeCovariance is indeed an estimator of eye pose
    covariance, but with extra terms. This happens because eye pose is nuisance
    parameter for the estimator of eye shape, and rather than marginalizing eye
    pose out out, we estimate it jointly with eye shape.
    """

    def __init__(self, derivative_data: derivative_calc.DataWrapper):
        """Initialize parameters of base class, plus some extra parameters.

        Args:
            derivative_data (derivative_calc.DataWrapper): data required for
            computation of derivatives.
        """

        # This initializes derivative calculators, covariance calculators for
        # inputs, number of visible glints, pupil points, and limbus points,
        # and member variables self.with_limbus as False (which is overwritten
        # with a new default), and self.d_optim_d_data and self.covariance,
        # both to None
        super().__init__(derivative_data)
        self.reset_derivatives()

        # List of default gaze angles used to collect eye-shape image features.
        self.list_gaze_angles_deg = \
            self.derivative_data.device.sample_fov([3, 3])

        # Whether to include limbus features in computations of eye-shape
        # covariance. By default, we do use the limbus in eye-shape
        # computations.
        self.with_limbus = True

    def load_input_covariances(
        self,
        parameter_file_name=os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            r"default_covariances/default_eye_shape_covariance_inputs.json"
        )
    ):
        """Set parameters for the computation of eye-shape covariances.

        Args:
            parameter_file_name (_type_, optional): name of file with inputs
            for computation of LEDs, camera, and feature covariances. Defaults
            to os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            "default_covariances/default_eye_shape_covariance_inputs.json" ).
        """

        with open(parameter_file_name) as parameter_file_stream:
            parameter_dict = json.load(parameter_file_stream)

        path = parameter_dict.get("path")
        if path is None:
            path = sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR

        common_inputs_covariance_file_name = \
            os.path.join(path, parameter_dict["common inputs"])
        # From base class.
        self._load_common_inputs(
            parameter_file_name=common_inputs_covariance_file_name
        )

        # Specific inputs to eye shape.
        gaze_inputs = parameter_dict["gaze inputs"]
        self.set_list_of_gaze_angles(gaze_inputs)
        self.with_limbus = parameter_dict["with limbus"]

    def set_list_of_gaze_angles(self, gaze_inputs):
        """Set list of gaze angles used to collect data for covariance comput.

        Args:
            gaze_inputs (list of int or list of torch.Tensor): if gaze_inputs
            is a list of integers, it must have size (2,) and it indicates the
            number of horizontal and vertical gaze directions to be sampled
            uniformly across the device's field of view; if gaze_inputs is a
            list of torch.Tensors, it directly corresponds to the gaze
            directions to be used to sample the device's field of view.
        """
        if isinstance(gaze_inputs[0], int):
            # Input may be the number of samples in horizontal and vertical
            # directions...
            self.list_gaze_angles_deg = \
                self.derivative_data.device.sample_fov(gaze_inputs)
        else:
            # Or directly the gaze directions that are sampled to generate
            # eye-shape covariance data.
            self.list_gaze_angles_deg = gaze_inputs

    def reset_derivatives(self):
        """Reset derivatives computed so far.
        """

        # These member variables accumulate values of the Hessian over multiple
        # gaze directions. The first index, always 1, indicates "current"
        # frame. Data from the next frame has an index of 2.
        self.H1_11 = 0
        self.H1_21 = \
            torch.empty(
                0, self.derivative_calculator_eye_shape.get_num_parameters()
            )
        self.H1_22 = torch.empty(0, 0)

        # These member variables accumulate values of the cross derivative over
        # multiple gaze directions. The first index, always 1, indicates
        # "current" frame. Data from the next frame has an index 2.
        num_rows = self.derivative_calculator_eye_shape.get_num_parameters()
        num_cols = \
            self.derivative_calculator_leds.get_num_parameters() + \
            self.derivative_calculator_camera_extrinsics.\
            get_num_parameters() + \
            self.derivative_calculator_camera_intrinsics.get_num_parameters()

        self.D1_11 = 0
        self.D1_12 = torch.empty(num_rows, 0)
        self.D1_1 = torch.empty(0, num_cols)
        self.D1_2 = torch.empty(0, 0)

        # List with the number of columns added to the partial derivative of
        # the optimization parameters with respect to the data for each new
        # image frame. This is variable, as there are different number of
        # glints, pupil points, and limbus points per frame.
        self.num_visible_glints = []
        self.num_visible_pupil_points = []
        self.num_visible_limbus_points = []

        # Once we reset the derivatives, these two variables are no longer
        # valid.
        self.d_optim_d_data = None
        self.covariance = None

    def compute_d_optim_d_data_indices(self):
        """Assemble a list of indices to selectively slice the Jacobian.

        The derivative of the optimization parameters has N columns
        sequentially split into K blocks, one for each image frame used for the
        computation of the derivatives, where the k-th block is subdivided
        into:

        * N1_k terms for LEDs; the subscript "k" can be dropped, as the number
          of LEDs is fixed.
        * N2_k terms for camera extrinsics; also fixed.
        * N3_k terms for camera intrinsics; also fixed.
        * N4_k terms for glint points; variable.
        * N5_k terms for pupil points; variable.
        * N6_k terms for limbus points; variable.

        For example, if a system with 12 LEDs and a camera model with 9
        intrinsic parameters is used to capture two image frames, and the
        frames have, respectively, 8 and 11 glint measurements, 22 and 25 pupil
        points, and 15 and 18 limbus points, the columns of the Jacobian will
        split as:

                   Frame 1               |              Frame 2
         LED | camera| glint-pupil-limbus|  LED | camera| glint-pupil-limbus
        3*12 | 6 | 9 | 2*8 | 2*22 | 2*15 | 3*12 | 6 | 9 | 2*11 | 2*25 | 2*18

        Returns:
            list of int: list of integers indexing components of the
            derivatives of the optimization parameters with respect to the
            data.
        """

        # Initialization.
        current_index = 0
        index_block = [current_index, ]

        # Contribution from LEDs.
        num_led_params = self.derivative_calculator_leds.get_num_parameters()
        current_index = current_index + num_led_params
        index_block = index_block + [current_index, ]

        # Contribution from extrinsic camera parameters.
        num_extrinsic_parameters = \
            self.derivative_calculator_camera_extrinsics.\
            get_num_parameters()
        current_index = current_index + num_extrinsic_parameters
        index_block = index_block + [current_index, ]

        # Contribution from intrinsic camera parameters.
        num_intrinsic_parameters = \
            self.derivative_calculator_camera_intrinsics.\
            get_num_parameters()
        current_index = current_index + num_intrinsic_parameters
        index_block = index_block + [current_index, ]

        index_list = index_block
        num_frames = len(self.num_visible_glints)
        for frame_idx in range(num_frames):
            index_block = []

            # Contribution from glints.
            num_glint_parameters = self.num_visible_glints[frame_idx] * 2
            current_index = current_index + num_glint_parameters
            index_block = index_block + [current_index, ]

            # Contribution from pupil.
            num_pupil_parameters = self.num_visible_pupil_points[frame_idx] * 2
            current_index = current_index + num_pupil_parameters
            index_block = index_block + [current_index, ]

            # Contribution from limbus.
            if self.with_limbus:
                num_limbus_parameters = \
                    self.num_visible_limbus_points[frame_idx] * 2
                current_index = current_index + num_limbus_parameters
                index_block = index_block + [current_index, ]

            index_list = index_list + index_block

        self.d_optim_d_data_indices = index_list

        return self.d_optim_d_data_indices

    def compute_d_optim_d_data(self):
        """Compute deriv. of eye-shape, pose and lifting params w.r.t. data.

        Args:
            list_gaze_rotation_deg (list of torch.Tensor): list of (2,) torch
            tensors corresponding to rotations to be applied to the nominal
            gaze direction.
        """

        for gaze_rotation_deg in self.list_gaze_angles_deg:
            # Rotate the eye to the desired gaze.
            self.derivative_data.rotate_eye(gaze_rotation_deg)

            # Generate the data for the gaze direction. We need to keep track
            # of how much data we have.
            self.prep_glint_data()
            self.num_visible_glints = \
                self.num_visible_glints + \
                [self.derivative_data.num_visible_glints, ]

            self.prep_pupil_data()
            self.num_visible_pupil_points = \
                self.num_visible_pupil_points + \
                [
                    self.derivative_calculator_pupil_lifting.
                    get_num_parameters(),
                ]

            if self.with_limbus:
                self.prep_limbus_data()
                self.num_visible_limbus_points = \
                    self.num_visible_limbus_points + \
                    [
                        self.derivative_calculator_limbus_lifting.
                        get_num_parameters(),
                    ]

            # Compute the Hessian and cross-derivative matrices.
            self._partial_compute_hessian()
            self._partial_compute_cross_derivative()

            # Unrotate the eye back to nominal.
            self.derivative_data.unrotate_eye(gaze_rotation_deg)

        hessian = \
            torch.vstack(
                (
                    torch.hstack((self.H1_11, self.H1_21.T)),  # type: ignore
                    torch.hstack((self.H1_21, self.H1_22))
                )
            )

        cross_derivative = \
            torch.vstack(
                (
                    torch.hstack((self.D1_11, self.D1_12)),  # type: ignore
                    torch.hstack((self.D1_1, self.D1_2))
                )
            )

        # The Hessian may be singular.
        self.d_optim_d_data = -torch.linalg.pinv(hessian) @ cross_derivative

        return self.d_optim_d_data

    @staticmethod
    def compute_covariance_standalone(
        d_shape_d_param,
        d_shape_d_param_indices,
        with_limbus,
        leds_covariance_calculator,
        camera_covariance_calculator,
        features_covariance_calculator
    ):
        """Static method for computation of eye-shape covariance.

        This is useful if the expensive derivatives of an EyeShapeCovariance
        object have already been computed and we wish to quickly change the
        inputs to the final computation of the covariance of eye-shape
        parameters.

        Args:
            d_shape_d_param (torch.Tensor): derivative of eye-shape
            optimization parameters with respect to input parameters.

            d_shape_d_param_indices (list of int): list with indices of LEDs,
            camera, and image feature components corresponding to the columns
            of d_optim_d_param.

            with_limbus (bool): whether limbus features were included in the
            computation of the derivative of eye-pose and lifting optimization
            parameters.

            leds_covariance_calculator (LEDsCovarianceCalculator): object used
            to compute the contribution of LEDs to the eye-pose covariance.

            camera_covariance_calculator (CameraCovarianceCalculator): object
            used to compute the contribution of cameras to the eye-pose
            covariance.

            features_covariance_calculator (FeaturesCovarianceCalculator):
            object used to compute the contribution of image features to the
            eye-pose covariance.

        Returns:
            torch.Tensor: covariance matrix of eye-shape parameters.
        """

        covariance = \
            leds_covariance_calculator.compute_covariance(
                "shape", d_shape_d_param, d_shape_d_param_indices
            )
        covariance = \
            covariance + \
            camera_covariance_calculator.compute_covariance(
                "shape", d_shape_d_param, d_shape_d_param_indices
            )
        covariance = \
            covariance + \
            features_covariance_calculator.compute_covariance(
                "shape", d_shape_d_param, d_shape_d_param_indices, with_limbus
            )

        return covariance

    def compute_covariance(self):
        """Compute the covariance of the eye-shape parameters.
        """

        if self.d_optim_d_data is None:
            self.compute_d_optim_d_data()
            self.compute_d_optim_d_data_indices()

        compute_inputs = self.leds_covariance_calculator is None
        compute_inputs = \
            compute_inputs or self.camera_covariance_calculator is None
        compute_inputs = \
            compute_inputs or self.features_covariance_calculator is None

        if compute_inputs:
            # This uses the default values of the input covariances. For
            # non-default values, self.load_input_covariances() must be
            # explicitly called with the appropriate configuration file as
            # input.
            self.load_input_covariances()

        num_shape_params = \
            self.derivative_calculator_eye_shape.get_num_parameters()
        self.covariance = self.compute_covariance_standalone(
            self.d_optim_d_data[:num_shape_params, :],
            self.d_optim_d_data_indices,
            self.with_limbus,
            self.leds_covariance_calculator,
            self.camera_covariance_calculator,
            self.features_covariance_calculator
        )

        return self.covariance

    def save_covariance(
        self,
        parameter_file_name=os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            r"default_covariances/default_eye_shape_covariances.json"
        )
    ):
        """Save eye-shape covariance as a json file.

        Using json.dump to output the results creates an ugly file.

        This output is used primarily as inputs to EyePoseCovariance objects
        for computation of eye-pose covariance from a know eye-shape
        covariance.
        """

        with open(parameter_file_name, "w") as parameter_file_stream:
            indent = 0

            # Increase indent after we write.
            parameter_file_stream.write(" " * indent + "{\n")
            indent += 2

            parameter_file_stream.write(" " * indent + "\"_comments\": [\n")
            indent += 2

            parameter_file_stream.write(
                " " * indent +
                "\"Eye-shape generated using eye_shape_covariance.py.\"\n"
            )

            # Close bracket for comment. Decrease indent before we close.
            indent -= 2
            parameter_file_stream.write(" " * indent + "],\n")

            # Extra space.
            parameter_file_stream.write("\n")

            parameter_file_stream.write(" " * indent + "\"cov\": [\n")
            indent += 2

            # Matrix is symmetric, so we can write rows or columns first.
            num_cols = len(self.covariance)
            for col_idx, column in enumerate(self.covariance):
                # First element is special, in that it is preceded by "[".
                parameter_file_stream.write(
                    " " * indent + "[{0:.9E}".format(column[0].item())
                )

                # Other elements are preceded by ", ".
                for item in column[1:]:
                    parameter_file_stream.write(
                        ", {0:.9E}".format(item.item())
                    )

                # No comma after the last bracket
                if col_idx < num_cols - 1:
                    parameter_file_stream.write("],\n")
                else:
                    parameter_file_stream.write("]\n")

            # Close square bracket for covariance-matrix data.
            indent -= 2
            parameter_file_stream.write(" " * indent + "]\n")

            # Close outermost curly brackets.
            indent -= 2
            parameter_file_stream.write(" " * indent + "}\n")

    def _partial_compute_hessian(self):
        """Compute the 2nd derv. cost function w.r.t to optim. parameters.
        """

        # This computes the derivative of eye pose as well as pupil (and
        # optionally limbus) lifting parameters.
        #
        # w_ = (w2 = eye_pose, w3 = pupil lifting, w4 = limbus lifting (opt.))
        #
        # H_partial = [d2_F_d_w__d_w_]

        H2_22 = super().compute_hessian()

        # The missing components are derivatives involving the eye-shape
        # parameters.
        #
        # We think of the optimization and data parameters w and z as.
        #
        # w = (w1 = eye shape, w2 = eye pose, w3 = pupil lifting, etc.)
        # z = (z2 = leds, z3 = camera extr., z4 = camera intr., etc)
        #
        # But for the eye-pose covariance calculator we have
        #
        # w = (w2 = eye pose, w3 = pupil lifting, etc.)
        # z = (z1 = eye shape, z2 = leds, z3 = camera extr., etc.)

        # This produces d2_F_d_(w2, w3, w4 (opt.))_d_z1
        H2_21 = super()._compute_cross_derivative_eye_shape()

        # And we must also compute d2_F_d_eye_shape_d_(eye_shape, angles).
        d2_cost_glints_d2_eye_shape = \
            self._compute_d2_cost_x_d2_eye_shape("glint")

        d2_cost_pupil_d2_eye_shape = \
            self._compute_d2_cost_x_d2_eye_shape("pupil")

        H2_11 = d2_cost_glints_d2_eye_shape + d2_cost_pupil_d2_eye_shape

        if self.with_limbus:
            d2_cost_limbus_d2_eye_shape = \
                self._compute_d2_cost_x_d2_eye_shape("limbus")
            H2_11 = H2_11 + d2_cost_limbus_d2_eye_shape

        # We compute the hessian in pieces because as we accumulate data over
        # multiple frames we add values to some pieces and stack values for
        # others.
        #
        # H1 = [ H1_11 |    H1_12 = H1_21.T]
        #      [ H1_21 | H1_22 = H1_partial],
        #
        #                         N1                  N2    <- columns
        #      [H1_11 + H2_11 | H1_12 |    H2_12 = H2_21.T]
        # H2 = [        H1_21 | H1_22 |                  0] N1 rows
        #      [        H2_21 |     0 | H2_22 = H2_partial] N2 rows

        N1 = self.H1_21.shape[0]
        N2 = H2_21.shape[0]

        self.H1_11 = self.H1_11 + H2_11
        self.H1_21 = torch.vstack((self.H1_21, H2_21))

        self.H1_22 = \
            torch.vstack(
                (
                    torch.hstack((self.H1_22, torch.zeros(N1, N2))),
                    torch.hstack((torch.zeros(N2, N1), H2_22))
                )
            )

    def _partial_compute_cross_derivative(self):
        """Compute 2nd derivative of cost w.r.t. optim. and input params.
        """

        # With optimize_eye_shape True, the cross derivative for the superclass
        # follow the pattern
        #
        # D_super = d_w_d_z,
        #
        # where
        #
        # w = (w2 = eye pose, w3 = pupil lift., w4 = limbus lift. (opt.))
        # z = (z2 = leds, z3 = camera extr., z4 = camera intr.,
        #           z5 = glints, z6 = pupil points, z7 = limbus points (opt.)).
        #
        # We start with index two because we reserve w1 and z1 for eye shape,
        # depending on the situation.
        #
        # To compute the cross derivative for the derived class, split the
        # derivative above:
        #
        # D1_super = [D1_1 D1_2],
        #
        # where the index 1 in D1 indicates the frame.
        #
        # The new derivative becomes
        #
        # D1 = [D1_11 = d_w1_d_z(2-4) | D1_12 = d_w1_d_z(5-7)]
        #      [             D1_1     |                  D1_2].
        #
        # This split reflects the fact that leds, camera extrinsics, and camera
        # intrinsics (z2 through z4) are fixed, whereas glints, pupil points,
        # and limbus points (z5 through z7) change from frame to frame
        #
        # As new frames are added, the derivative grows:
        #
        #                 N0      N1     N2    <- # columns
        #      [D1_11 + D2_11 | D1_12 | D2_12]
        # D2 = [         D1_1 |  D1_2 |     0] M1 rows
        #      [         D2_1 |     0 |  D2_2] M2 rows
        #
        # We output each piece so we can assemble the final cross derivative by
        # accumulating data over multiple frames.

        D_super = super().compute_cross_derivative(optimize_eye_shape=True)
        N0 = \
            self.derivative_calculator_leds.get_num_parameters() + \
            self.derivative_calculator_camera_extrinsics.get_num_parameters() \
            + \
            self.derivative_calculator_camera_intrinsics.get_num_parameters()

        # Split the cross derivative of the super-class into fixed (leds,
        # camera) and varying (image measurements) parameters.
        D2_1, D2_2 = torch.tensor_split(D_super, [N0, ], dim=1)

        D2_11, D2_12 = self._compute_d2_cost_d_eye_shape_d_data()

        self.D1_11 = self.D1_11 + D2_11

        M1 = self.D1_1.shape[0]
        M2 = D2_1.shape[0]
        N1 = self.D1_12.shape[1]
        N2 = D2_12.shape[1]

        self.D1_12 = torch.hstack((self.D1_12, D2_12))
        self.D1_1 = torch.vstack((self.D1_1, D2_1))
        self.D1_2 = \
            torch.vstack(
                (
                    torch.hstack((self.D1_2, torch.zeros(M1, N2))),
                    torch.hstack((torch.zeros(M2, N1), D2_2))
                )
            )

    def _compute_d2_cost_x_d2_eye_shape(self, feature_name):
        """Compute 2nd deriv. of feature cost w.r.t. to eye-shape params.

        In the method's name and implementation "x" is a stand in for "glint,"
        "pupil" or "limbus".

        Args:
            feature_name (str): One of "glint," "pupil" or "limbus" depending
            on which component of the cost function is considered.
        """

        feature_name = feature_name.lower()
        if feature_name == "glint":
            d_all_x_d_eye_shape = self.d_all_glints_d_eye_shape
        elif feature_name == "pupil":
            d_all_x_d_eye_shape = self.d_pupil_d_eye_shape
        else:
            d_all_x_d_eye_shape = self.d_limbus_d_eye_shape

        d2_cost_x_d2_eye_shape = 0

        for d_x_d_eye_shape in d_all_x_d_eye_shape:
            if d_x_d_eye_shape is None:
                # Some pupil/limbus points are not visible.
                continue

            d2_cost_x_d2_eye_shape = \
                d2_cost_x_d2_eye_shape + d_x_d_eye_shape.T @ d_x_d_eye_shape

        return d2_cost_x_d2_eye_shape

    def _compute_d2_cost_x_d_eye_shape_d_data(self, feature_name):
        """Compute deriv. of pupil/limbus errors w.r.t. eye shape and data.

        In the method's name and implementation "x" is a stand in for "pupil"
        or "limbus".

        Args:
            feature_name (str): One of "pupil" or "limbus" depending on which
            component of the cost is considered.
        """

        if feature_name.lower() == "pupil":
            # Optimization parameters
            d_all_x_d_eye_shape = self.d_pupil_d_eye_shape

            # Data
            d_all_x_d_extrinsics = self.d_pupil_d_camera_extrinsics
            d_all_x_d_intrinsics = self.d_pupil_d_camera_intrinsics
        else:
            # Optimization parameters
            d_all_x_d_eye_shape = self.d_limbus_d_eye_shape

            # Data
            d_all_x_d_extrinsics = self.d_limbus_d_camera_extrinsics
            d_all_x_d_intrinsics = self.d_limbus_d_camera_intrinsics

        d2_cost_x_d_eye_shape_d_extrinsics = 0
        d2_cost_x_d_eye_shape_d_intrinsics = 0

        for i in range(len(d_all_x_d_eye_shape)):
            d_x_d_eye_shape = d_all_x_d_eye_shape[i]
            if d_x_d_eye_shape is None:
                # Pupil or limbus point is not visible.
                continue

            d_x_d_extrinsics = d_all_x_d_extrinsics[i]
            d_x_d_intrinsics = d_all_x_d_intrinsics[i]

            d2_cost_x_d_eye_shape_d_extrinsics = \
                d2_cost_x_d_eye_shape_d_extrinsics + \
                d_x_d_eye_shape.T @ d_x_d_extrinsics

            d2_cost_x_d_eye_shape_d_intrinsics = \
                d2_cost_x_d_eye_shape_d_intrinsics + \
                d_x_d_eye_shape.T @ d_x_d_intrinsics

        # Special parameters. These are the derivative with respect to
        # pupil/limbus measurements of the transpose of the derivative of the
        # pupil/limbus error with respect to eye shape. This derivative has
        # size (dim optimization parameter) x (2 x # visible pupil/limbus
        # points), where the factor two comes from the fact that pupil/limbus
        # points are two dimensional.
        d_all_visible_x_d_eye_shape = \
            [d for d in d_all_x_d_eye_shape if d is not None]
        d2_cost_x_d_eye_shape_d_x_measurements = \
            -1 * torch.vstack(d_all_visible_x_d_eye_shape).T

        return \
            d2_cost_x_d_eye_shape_d_extrinsics, \
            d2_cost_x_d_eye_shape_d_intrinsics, \
            d2_cost_x_d_eye_shape_d_x_measurements

    def _compute_d2_cost_d_eye_shape_d_data(self):
        """Compute the derivative of the cost w.r.t. eye shape and data.
        """

        d2_glint_cost_d_eye_shape_d_leds, \
            d2_glint_cost_d_eye_shape_d_extrinsics, \
            d2_glint_cost_d_eye_shape_d_intrinsics, \
            d2_glint_cost_d_eye_shape_d_glint_measurements = \
            self._compute_d2_cost_glints_d_optim_d_data(
                self.d_all_glints_d_eye_shape
            )

        d2_pupil_cost_d_eye_shape_d_extrinsics, \
            d2_pupil_cost_d_eye_shape_d_intrinsics, \
            d2_pupil_cost_d_eye_shape_d_pupil_measurements = \
            self._compute_d2_cost_x_d_eye_shape_d_data("pupil")

        d2_F_d_w1_d_z2 = d2_glint_cost_d_eye_shape_d_leds
        d2_F_d_w1_d_z3 = \
            d2_glint_cost_d_eye_shape_d_extrinsics + \
            d2_pupil_cost_d_eye_shape_d_extrinsics
        d2_F_d_w1_d_z4 = \
            d2_glint_cost_d_eye_shape_d_intrinsics + \
            d2_pupil_cost_d_eye_shape_d_intrinsics
        d2_F_d_w1_d_z5 = d2_glint_cost_d_eye_shape_d_glint_measurements
        d2_F_d_w1_d_z6 = d2_pupil_cost_d_eye_shape_d_pupil_measurements

        if self.with_limbus:
            d2_limbus_cost_d_eye_shape_d_extrinsics, \
                d2_limbus_cost_d_eye_shape_d_intrinsics, \
                d2_limbus_cost_d_eye_shape_d_limbus_measurements = \
                self._compute_d2_cost_x_d_eye_shape_d_data("limbus")

            d2_F_d_w1_d_z3 = \
                d2_F_d_w1_d_z3 + d2_limbus_cost_d_eye_shape_d_extrinsics
            d2_F_d_w1_d_z4 = \
                d2_F_d_w1_d_z4 + d2_limbus_cost_d_eye_shape_d_intrinsics
            d2_F_d_w1_d_z7 = d2_limbus_cost_d_eye_shape_d_limbus_measurements
        else:
            num_eye_shape_params = \
                self.derivative_calculator_eye_shape.get_num_parameters()
            d2_F_d_w1_d_z7 = torch.empty(num_eye_shape_params, 0)

        # We return result in blocks so that the cross derivative can be
        # accumulated over multiple frames.
        d2_F_d_w1_d_z2_to_4 = \
            torch.hstack(
                (
                    d2_F_d_w1_d_z2,
                    d2_F_d_w1_d_z3,
                    d2_F_d_w1_d_z4
                )  # type: ignore
            )
        d2_F_d_w1_d_z5_to_7 = \
            torch.hstack(
                (
                    d2_F_d_w1_d_z5,
                    d2_F_d_w1_d_z6,
                    d2_F_d_w1_d_z7
                )  # type: ignore
            )

        return d2_F_d_w1_d_z2_to_4, d2_F_d_w1_d_z5_to_7
