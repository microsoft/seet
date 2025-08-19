"""Class and functions for computing covariance matrix of eye-pose parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import seet.core as core
import seet.sensitivity_analysis.camera_covariance_calculator as \
    camera_covariance_calculator
import seet.sensitivity_analysis.derivative_calculators as derivative_calc
import seet.sensitivity_analysis.features_covariance_calculator as \
    feature_covariance_calculator
import seet.sensitivity_analysis.leds_covariance_calculator as \
    leds_covariance_calculator
import seet.sensitivity_analysis.sensitivity_analysis_configs as \
    sensitivity_analysis_configs
import os
import torch


class EyePoseCovariance():
    """Class and for computing covariance matrix of eye-pose parameters.
    """

    def __init__(self, derivative_data: derivative_calc.DataWrapper):
        """Initialize parameters for computation of eye-pose covariance.

        Args:
            derivative_data (derivative_calc.DataWrapper): data required for
            computation of derivatives.
        """

        self.derivative_data = derivative_data

        # These are used to compute the Hessian of the cost function with
        # respect to optimization parameters.
        self.derivative_calculator_eye_pose = \
            derivative_calc.EyePoseDerivatives(self.derivative_data)

        self.derivative_calculator_pupil_lifting = \
            derivative_calc.PupilLiftingDerivatives(self.derivative_data)

        self.derivative_calculator_limbus_lifting = \
            derivative_calc.LimbusLiftingDerivatives(self.derivative_data)

        # These are used to compute the second derivative of the cost function
        # with respect to data parameters.
        self.derivative_calculator_camera_extrinsics = \
            derivative_calc.CameraExtrinsicsDerivatives(self.derivative_data)

        self.derivative_calculator_camera_intrinsics = \
            derivative_calc.CameraIntrinsicsDerivatives(self.derivative_data)

        self.derivative_calculator_leds = \
            derivative_calc.LEDLocationsDerivatives(self.derivative_data)

        # Eye shape is special in that it can be either an optimization (during
        # user calibration) or a data parameter (during eye tracking)
        self.derivative_calculator_eye_shape = \
            derivative_calc.EyeShapeDerivatives(self.derivative_data)

        # Covariance calculators for inputs.
        self.leds_covariance_calculator = None
        self.camera_covariance_calculator = None
        self.features_covariance_calculator = None

        self.num_visible_glints = 0
        self.num_visible_pupil_points = 0
        self.num_visible_limbus_points = 0

        # Whether to include limbus features in computations of eye-pose
        # covariance. By default, we do not use the limbus in eye-shape
        # computations.
        self.with_limbus = False

        # Derivative of eye parameters with respect to inputs.
        self.d_optim_d_data = None

        # Eye-pose covariance is needed to compute eye-shape covariance.
        self.eye_shape_covariance = None

        # Eye-shape covariance.
        self.covariance = None

    def _load_common_inputs(
        self,
        parameter_file_name=os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            r"default_covariances/default_common_inputs.json"
        )
    ):
        """Set inputs for computing eye-pose cov., excluding eye-shape params.

        Args:
            parameter_file_name (_type_, optional): name of file with inputs
            for computation of LEDs, camera, and feature covariances. Defaults
            to os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            "default_covariances/default_common_inputs.json" ).
        """
        with open(parameter_file_name) as parameter_file_stream:
            parameter_dict = json.load(parameter_file_stream)

        path = parameter_dict.get("path")
        if path is None:
            path = sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR

        leds_covariance_file_name = os.path.join(path, parameter_dict["leds"])
        self.leds_covariance_calculator = \
            leds_covariance_calculator.LEDsCovarianceCalculator(
                parameter_file_name=leds_covariance_file_name
            )

        camera_covariance_file_name = \
            os.path.join(path, parameter_dict["camera"])
        self.camera_covariance_calculator = \
            camera_covariance_calculator.CameraCovarianceCalculator(
                parameter_file_name=camera_covariance_file_name
            )

        features_covariance_file_name = \
            os.path.join(path, parameter_dict["features"])
        self.features_covariance_calculator = \
            feature_covariance_calculator.FeaturesCovarianceCalculator(
                parameter_file_name=features_covariance_file_name
            )

    def load_input_covariances(
        self,
        parameter_file_name=os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            r"default_covariances/default_eye_pose_covariance_inputs.json"
        ),
        eye_shape_covariance=None
    ):
        """Set parameters for the computation of eye-pose covariances.

        Args:
            parameter_file_name (str, optional): name of file with inputs for
            computation of eye-shape, LEDs, camera, and feature covariances.
            Defaults to os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            "default_covariances/default_eye_pose_covariance_inputs.json" ).

            eye_shape_covariances (torch.Tensor, optional): if not None, this
            is the eye-shape covariance matrix required for the computation of
            eye-shape covariances. If absent, the eye-shape covariance is read
            from the input file.
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

        if eye_shape_covariance is None:
            eye_shape_covariance_file_name = \
                os.path.join(path, parameter_dict["eye-shape covariance"])
            self.load_eye_shape_covariance(
                parameter_file_name=eye_shape_covariance_file_name
            )
        else:
            self.eye_shape_covariance = eye_shape_covariance

        self.with_limbus = parameter_dict["with limbus"]

    def load_eye_shape_covariance(
        self,
        parameter_file_name=os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            r"default_covariances/default_eye_shape_covariances.json"
        )
    ):
        """Set input eye-shape covs. for computation of eye-pose covariances.

        Args:
            parameter_file_name (str, optional): name of file with eye-pose
            covariance matrix. Defaults to os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            "default_covariances/default_eye_shape_covariances.json" ).
        """
        with open(parameter_file_name) as parameter_file_stream:
            covariance_dict = json.load(parameter_file_stream)

        self.eye_shape_covariance = torch.tensor(covariance_dict["cov"])

    def prep_glint_data(self):
        """Generate glint data and its derivatives.
        """

        self.derivative_data.generate_glints_inPixels()

        # Optimization parameters.
        self.d_all_glints_d_eye_pose = \
            self.derivative_calculator_eye_pose.compute_d_glints_d_parameters()

        # Data parameters.
        self.d_all_glints_d_leds = \
            self.derivative_calculator_leds.compute_d_glints_d_parameters()
        self.d_all_glints_d_camera_extrinsics = \
            self.derivative_calculator_camera_extrinsics.\
            compute_d_glints_d_parameters()
        self.d_all_glints_d_camera_intrinsics = \
            self.derivative_calculator_camera_intrinsics.\
            compute_d_glints_d_parameters()

        # Sometimes optimization, sometimes data.
        self.d_all_glints_d_eye_shape = \
            self.derivative_calculator_eye_shape.\
            compute_d_glints_d_parameters()

    def prep_pupil_data(self):
        """Generate pupil data and its derivatives.
        """

        self.derivative_data.generate_pupil_inPixels()

        # Optimization parameters.
        self.d_pupil_d_eye_pose = \
            self.derivative_calculator_eye_pose.compute_d_pupil_d_parameters()
        self.d_pupil_d_angles = \
            self.derivative_calculator_pupil_lifting.\
            compute_d_pupil_d_parameters()

        # Data parameters.
        self.d_pupil_d_camera_extrinsics = \
            self.derivative_calculator_camera_extrinsics.\
            compute_d_pupil_d_parameters()
        self.d_pupil_d_camera_intrinsics = \
            self.derivative_calculator_camera_intrinsics.\
            compute_d_pupil_d_parameters()

        # Sometimes optimization, sometimes data.
        self.d_pupil_d_eye_shape = \
            self.derivative_calculator_eye_shape.compute_d_pupil_d_parameters()

    def prep_limbus_data(self):
        """Generate limbus data and its derivatives.
        """

        self.derivative_data.generate_limbus_inPixels()

        # Optimization parameters.
        self.d_limbus_d_eye_pose = \
            self.derivative_calculator_eye_pose.compute_d_limbus_d_parameters()
        self.d_limbus_d_angles = \
            self.derivative_calculator_limbus_lifting.\
            compute_d_limbus_d_parameters()

        # Data parameters.
        self.d_limbus_d_camera_extrinsics = \
            self.derivative_calculator_camera_extrinsics.\
            compute_d_limbus_d_parameters()
        self.d_limbus_d_camera_intrinsics = \
            self.derivative_calculator_camera_intrinsics.\
            compute_d_limbus_d_parameters()

        # Sometimes optimization, sometimes data.
        self.d_limbus_d_eye_shape = \
            self.derivative_calculator_eye_shape.\
            compute_d_limbus_d_parameters()

    def compute_hessian(self):
        """Compute the Hessian of the cost function.

        Returns:
            torch.tensor: Hessian of the cost function.
        """

        d2_glint_d2_eye_pose = self._compute_d2_cost_glints_d2_eye_pose()
        d2_pupil_d2_eye_pose, \
            d2_pupil_d2_angles, \
            d2_pupil_d_eye_pose_d_angles = \
            self._compute_d2_cost_x_d2_optimization("pupil")

        # Building up of the Hessian matrix H. We start with
        #
        #     [    top_left    top_middle]   [    top_left middle]
        # H = [top_middle.T bottom_middle] = [top_middle.T middle]

        # Top left has contributions from glints and pupil.
        top_left = d2_glint_d2_eye_pose + d2_pupil_d2_eye_pose
        top_middle = d2_pupil_d_eye_pose_d_angles

        bottom_middle = d2_pupil_d2_angles
        middle = torch.vstack((top_middle, bottom_middle))  # type: ignore

        if self.with_limbus:
            d2_limbus_d2_eye_pose, \
                d2_limbus_d2_angles, \
                d2_limbus_d_eye_pose_d_angles = \
                self._compute_d2_cost_x_d2_optimization("limbus")

            # Top left has contribution from limbus
            top_left = top_left + d2_limbus_d2_eye_pose

            # Since we have limbus, we grow the matrix.
            #
            #     [    top_left middle        top_right]
            # H = [top_middle.T middle middle_right = 0]
            #
            #     [    top_left middle right]
            #   = [top_middle.T middle right]

            top_right = d2_limbus_d_eye_pose_d_angles
            num_pupil_params = \
                self.derivative_calculator_pupil_lifting.get_num_parameters()
            num_limbus_params = \
                self.derivative_calculator_limbus_lifting.get_num_parameters()
            middle_right = torch.zeros(num_pupil_params, num_limbus_params)

            # Keep growing...
            #
            #     [    top_left   middle        right]
            # H = [top_middle.T   middle        right] [      (right right).T
            #     bottom_right]
            #
            #     [    top_left middle  right]
            #   = [top_middle.T middle  right] [      bottom bottom bottom]
            right = torch.vstack((top_right, middle_right))  # type: ignore

            bottom_right = d2_limbus_d2_angles
            bottom = torch.hstack((right.T, bottom_right))  # type: ignore
        else:
            num_pose_plus_pupil_angles = \
                self.derivative_calculator_eye_pose.get_num_parameters() + \
                self.derivative_calculator_pupil_lifting.get_num_parameters()
            right = torch.zeros(num_pose_plus_pupil_angles, 0)
            bottom = right.T

        # Final assembly.
        #
        # H = [left   middle  right] [bottom bottom bottom]
        #
        #   = [   top] [bottom]
        left = torch.vstack((top_left, top_middle.T))  # type: ignore
        top = torch.hstack((left, middle, right))  # type: ignore

        return torch.vstack((top, bottom))

    def compute_cross_derivative(self, optimize_eye_shape=False):
        """Computes the second derivative of the cost w.r.t. optim and data.

        Args:
            optimize_eye_shape (bool, optional): if True, eye shape is
            considered an optimization parameter. That is typically the case
            when performing user calibration. However, during eye tracking eye
            shape is considered a data parameter. Defaults to False.

        Returns:
            torch.Tensor: torch tensor corresponding to the second derivative
            of the cost with respect to optimization and data parameters.
        """

        # Let the cost be denoted by F. It has up to three components:
        #
        # F = F1 (glints) + F2 (pupil) + F3 (limbus, optional)
        #
        # The optimization parameters are
        #
        # w2 = eye pose w3 = pupil lifting parameters w4 = limbus lifting
        # parameters (optional)
        #
        # The data parameters are
        #
        # z1 = eye shape (optional) z2 = leds z3 = camera extrinsics z4 =
        # camera intrinsics z5 = glint measurements z6 = pupil measurements z7
        # = limbus measurements (optional)
        #
        # The cross derivative matrix is d2_F_d_w_d_z, with blocks given by
        # d2_F_d_wi_dzj so that
        #
        #                [d2_F_d_w2_d_z1 d2_F_d_w2_d_z2 ... d2_F_d_w2_d_z7]
        # d2_F_d_w_d_z = [d2_F_d_w3_d_z1 d2_F_d_w3_d_z2 ... d2_F_d_w3_d_z7]
        #                [d2_F_d_w4_d_z1 d2_F_d_w4_d_z2 ... d2_F_d_w4_d_z7]
        #
        # More explicitly:
        #
        # d2_F1_d_w2_d_z1 = d2_glint_cost_d_eye_pose_d_eye_shape,
        # d2_F1_d_w2_d_z2 = d2_glint_cost_d_eye_pose_d_leds,
        #
        # etc.

        d2_glint_cost_d_eye_pose_d_leds, \
            d2_glint_cost_d_eye_pose_d_extrinsics, \
            d2_glint_cost_d_eye_pose_d_intrinsics, \
            d2_glint_cost_d_eye_pose_d_glint_measurements = \
            self._compute_d2_cost_glints_d_optim_d_data(
                self.d_all_glints_d_eye_pose
            )

        d2_pupil_cost_d_eye_pose_d_extrinsics, \
            d2_pupil_cost_d_eye_pose_d_intrinsics, \
            d2_pupil_cost_d_eye_pose_d_pupil_measurements, \
            d2_pupil_cost_d_angles_d_extrinsics, \
            d2_pupil_cost_d_angles_d_intrinsics, \
            d2_pupil_cost_d_angles_d_pupil_measurements = \
            self._compute_d2_cost_x_d_optim_d_data("pupil")

        # We start from derivatives with respect to z2 because z1 is optional.
        d2_F_d_w2_d_z2 = d2_glint_cost_d_eye_pose_d_leds
        d2_F_d_w2_d_z3 = \
            d2_glint_cost_d_eye_pose_d_extrinsics + \
            d2_pupil_cost_d_eye_pose_d_extrinsics
        d2_F_d_w2_d_z4 = \
            d2_glint_cost_d_eye_pose_d_intrinsics + \
            d2_pupil_cost_d_eye_pose_d_intrinsics
        d2_F_d_w2_d_z5 = d2_glint_cost_d_eye_pose_d_glint_measurements
        d2_F_d_w2_d_z6 = d2_pupil_cost_d_eye_pose_d_pupil_measurements
        # d2_F2_d_w2_d_z7 = zero and optional, take care of it later.

        num_led_parameters = \
            self.derivative_calculator_leds.get_num_parameters()
        num_pupil_lifting_angles = \
            self.derivative_calculator_pupil_lifting.get_num_parameters()
        num_glint_measurements = d2_F_d_w2_d_z5.shape[1]

        d2_F_d_w3_d_z2 = \
            torch.zeros(num_pupil_lifting_angles, num_led_parameters)
        d2_F_d_w3_d_z3 = d2_pupil_cost_d_angles_d_extrinsics
        d2_F_d_w3_d_z4 = d2_pupil_cost_d_angles_d_intrinsics

        d2_F_d_w3_d_z5 = \
            torch.zeros(num_pupil_lifting_angles, num_glint_measurements)
        d2_F_d_w3_d_z6 = d2_pupil_cost_d_angles_d_pupil_measurements

        # Build a matrix without the optional components.
        d2_F_d_optim_d_z2 = \
            torch.vstack((d2_F_d_w2_d_z2, d2_F_d_w3_d_z2))  # type: ignore
        d2_F_d_optim_d_z3 = \
            torch.vstack((d2_F_d_w2_d_z3, d2_F_d_w3_d_z3))  # type: ignore
        d2_F_d_optim_d_z4 = \
            torch.vstack((d2_F_d_w2_d_z4, d2_F_d_w3_d_z4))  # type: ignore
        d2_F_d_optim_d_z5 = \
            torch.vstack((d2_F_d_w2_d_z5, d2_F_d_w3_d_z5))  # type: ignore
        d2_F_d_optim_d_z6 = \
            torch.vstack((d2_F_d_w2_d_z6, d2_F_d_w3_d_z6))  # type: ignore
        d2_F_d_optim_d_data = \
            torch.hstack(
                (
                    d2_F_d_optim_d_z2,
                    d2_F_d_optim_d_z3,
                    d2_F_d_optim_d_z4,
                    d2_F_d_optim_d_z5,
                    d2_F_d_optim_d_z6
                )
            )

        if self.with_limbus:
            num_visible_pupil_points = num_pupil_lifting_angles

            d2_F3_d_optim_d_data, \
                d2_F3_d_w4_d_data, \
                d2_F3_d_optim_d_z7 = \
                self._compute_cross_derivative_limbus_terms(
                    num_led_parameters,
                    num_glint_measurements,
                    num_visible_pupil_points
                )

            # Add the contribution to the top-left 2 x 2 block of blocks.
            d2_F_d_optim_d_data = d2_F_d_optim_d_data + d2_F3_d_optim_d_data

            # Augment the rows with derivative with respect to limbus lifting
            # parameters.
            d2_F_d_optim_d_data = \
                torch.vstack((d2_F_d_optim_d_data, d2_F3_d_w4_d_data))

            # Augment the columns with derivatives with respect to limbus
            # measurements.
            d2_F_d_optim_d_data = \
                torch.hstack((d2_F_d_optim_d_data, d2_F3_d_optim_d_z7))

        if optimize_eye_shape is False:
            # If eye shape is an not optimization parameter, it is a data
            # parameter and should contribute to increasing the columns of the
            # cross derivative.
            d2_F_d_optim_d_z1 = self._compute_cross_derivative_eye_shape()
            d2_F_d_optim_d_data = \
                torch.hstack((d2_F_d_optim_d_z1, d2_F_d_optim_d_data))
        else:
            # This is here so that we can add the comment. If eye shape is an
            # optimization parameter, it contributes to increasing the rows of
            # the cross derivative. However, we don't do this here, we leave it
            # to the EyeShapeCovariance class to handle what to do then.
            pass

        return d2_F_d_optim_d_data

    def compute_d_optim_d_data_indices(self):
        """Assemble a list of indices to selectively slice the Jacobian.
        """

        # Initialization.
        current_index = 0
        index_block = [current_index, ]

        # Contribution from eye-shape parameters.
        num_eye_shape_parameters = \
            self.derivative_calculator_eye_shape.get_num_parameters()
        current_index = current_index + num_eye_shape_parameters
        index_block = index_block + [current_index, ]

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

        # Contribution from glints.
        num_glint_parameters = self.num_visible_glints * 2
        current_index = current_index + num_glint_parameters
        index_block = index_block + [current_index, ]

        # Contribution from pupil.
        num_pupil_parameters = self.num_visible_pupil_points * 2
        current_index = current_index + num_pupil_parameters
        index_block = index_block + [current_index, ]

        # Contribution from limbus.
        if self.with_limbus:
            num_limbus_parameters = self.num_visible_limbus_points * 2
            current_index = current_index + num_limbus_parameters
            index_block = index_block + [current_index, ]

        self.d_optim_d_data_indices = index_block

        return self.d_optim_d_data_indices

    def compute_d_optim_d_data(self):
        """Compute the derivative of eye pose and lifting params w.r.t. data.

        For the purpose of computing the covariance of eye pose, eye shape is a
        data parameter. The other data parameters are camera extrinsic and
        intrinsics, led locations, and glint, pupil and, optionally, limbus
        measurements.

        Args:
            with_limbus (bool, optional): if True, limbus measurements are
            available. This is not typical for eye tracking. Defaults to False.
        """

        # Generate the data and keep track of how much data was used.
        self.prep_glint_data()
        self.num_visible_glints = self.derivative_data.num_visible_glints

        self.prep_pupil_data()
        self.num_visible_pupil_points = \
            self.derivative_calculator_pupil_lifting.get_num_parameters()

        if self.with_limbus:
            self.prep_limbus_data()
            self.num_visible_limbus_points = \
                self.derivative_calculator_limbus_lifting.get_num_parameters()

        hessian = self.compute_hessian()
        cross_derivative = \
            self.compute_cross_derivative(optimize_eye_shape=False)

        # The Hessian may be singular.
        self.d_optim_d_data = -torch.linalg.pinv(hessian) @ cross_derivative

        return self.d_optim_d_data

    @staticmethod
    def compute_covariance_standalone(
        d_pose_d_param,
        d_pose_d_param_indices,
        with_limbus,
        eye_shape_covariance,
        leds_covariance_calculator,
        camera_covariance_calculator,
        features_covariance_calculator
    ):
        """Low-level static method for computing cov. of eye-pose parameters.

        This method allows us to re-use derivatives computed with an
        EyePoseCovariance object with different input covariances for
        eye-shape, LEDs, cameras and image features.

        Args:
            d_pose_d_param (torch.Tensor): derivative of eye-pose optimization
            parameters with respect to input parameters.

            d_pose_d_param_indices (list of int): list with indices of
            eye-shape, LEDs, camera, and image feature components corresponding
            to the columns of d_optim_d_param.

            with_limbus (bool): whether limbus features were included in the
            computation of the derivative of eye-pose and lifting optimization
            parameters.

            eye_shape_covariance (torch.Tensor): covariance matrix of eye-shape
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
            torch.Tensor: covariance matrix of eye-pose parameters.
        """
        # [0, num_eye_shape_params, num_led_params, num_extr_params, etc.]
        num_eye_shape_params = d_pose_d_param_indices[1]
        D = d_pose_d_param[:, :num_eye_shape_params]

        covariance = D @ eye_shape_covariance @ D.T
        covariance = \
            covariance + \
            leds_covariance_calculator.compute_covariance(
                "pose", d_pose_d_param, d_pose_d_param_indices
            )
        covariance = \
            covariance + \
            camera_covariance_calculator.compute_covariance(
                "pose", d_pose_d_param, d_pose_d_param_indices
            )
        covariance = \
            covariance + \
            features_covariance_calculator.compute_covariance(
                "pose", d_pose_d_param, d_pose_d_param_indices, with_limbus
            )

        return covariance

    def compute_covariance(self):
        """Compute the covariance of the eye pose.
        """

        # d_optim_d_data is a
        #
        # (N2 + N3 + N4) x (M1 + M2 + M3 + M4 + M5 + M6 + M7)
        #
        # matrix, where
        #
        # N2: number of eye-pose parameters N3: number of pupil lifting
        # parameters N4: number of limbus lifting parameters
        #
        # and
        #
        # M1: number of eye-shape parameters M2: number of led parameters M3:
        # number of camera extrinsic parameters M4: number of camera intrinsic
        # parameters M5: 2 x number of glints M6: 2 x number of pupil points
        # M7: 2 x number of limbus points
        if self.d_optim_d_data is None:
            self.compute_d_optim_d_data()
            self.compute_d_optim_d_data_indices()

        if self.eye_shape_covariance is None:
            # This uses the default values of the eye-shape covariances. For
            # non-default values, self.load_eye_shape_covariance() must be
            # explicitly called with the appropriate configuration file as
            # input.
            self.load_eye_shape_covariance()

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
            self._load_common_inputs()

        num_pose_params = \
            self.derivative_calculator_eye_pose.get_num_parameters()
        self.covariance = \
            self.compute_covariance_standalone(
                self.d_optim_d_data[:num_pose_params, :],
                self.d_optim_d_data_indices,
                self.with_limbus,
                self.eye_shape_covariance,
                self.leds_covariance_calculator,
                self.camera_covariance_calculator,
                self.features_covariance_calculator
            )

        return self.covariance

    def compute_variance_gaze_error(self):
        """Compute the variance of the error in the gaze angle, in degrees.

        Compute the variance of gaze error given the eye-pose covariance. The
        eye pose covariance is given with respect to parameters

        theta_1: horizontal gaze [deg] (first rotation applied to nominal gaze)
        theta_2: vertical gaze [deg] (second rotation applied to nominal gaze)
        x, y, z: position of center of rotation of eye [mm].
        """

        # This formula is derived at https://mixedrealitywiki.com/x/tDJxD. The
        # approximation is valid for a virtual display at a large distance
        # (e.g., 1 m) and relatively small error in eye position (e.g., 5 mm).
        return torch.diag(self.covariance[:2, :2]).sum()

    @staticmethod
    def compute_covariance_pupil_center_standalone(
        d_pupil_center_d_pose_and_dist,
        eye_pose_covariance,
        eye_shape_covariance
    ):
        """Compute the covariance of the error in the pupil center.

        Args:
            d_pupil_center_d_pose_and_dist (torch.Tensor): (3, 6) torch tensor
            containing the derivative of the pupil center in mm with respect to
            the 2 gaze-angle parameters in deg, the 3 eye-position parameters
            in mm, and the distance from the eye rotation center to the pupil
            plane in mm.

            eye_pose_covariance (torch.Tensor): (5, 5) torch tensor holding the
            covariance of the eye-pose parameters: horiz. gaze angle, vert.
            gaze angle, and x, y, and z eye position.

            eye_shape_covariance (torch.Tensor): torch tensor holding the
            covariance of the eye-shape parameters.
        """

        d_pupil_d_pose = d_pupil_center_d_pose_and_dist[:, :-1]
        d_pupil_d_dist = d_pupil_center_d_pose_and_dist[:, -1]

        # The eye shape is represented as
        #
        # 0: dist_from_rot_center_to_cornea_center 1:
        # dist_from_rot_center_to_pupil_plane <-- this is dist 2:
        # dist_from_rot_center_to_limbus_plane 3-5: cornea shape parameters 6:
        # refractive_index 7: pupil_radius 8: limbus_radius
        cov_dist = eye_shape_covariance[1, 1]

        eye_pose_component = \
            d_pupil_d_pose @ eye_pose_covariance @ d_pupil_d_pose.T

        eye_shape_component = \
            torch.outer(d_pupil_d_dist, d_pupil_d_dist) * cov_dist

        return eye_pose_component + eye_shape_component

    def compute_covariance_pupil_center(self, node=None):
        """Compute the covariance of the error in the pupil center.

        Args:
            node (NodeModel, optional): secrets node object in which x, y, and z
            errors are expressed. Defaults to None, in which case the user
            coordinate system is used.
        """

        # The pupil center x is given by
        #
        # x = n * dist + c,
        #
        # where n is the unit gaze direction, dist is the distance between the
        # pupil center and the rotation center, and c is the rotation center.
        # Moreover, n = R_x(vert. angle) * R_y(horiz. angle). Therefore,
        #
        # d_x_d_(angles, c, dist) = (d_n_d_angles * dist, eye(3), n)

        n = \
            self.derivative_data.eye.get_gaze_direction_inOther(
                self.derivative_data.user
            )
        dist = \
            self.derivative_data.eye.\
            distance_from_rotation_center_to_pupil_plane

        d_n_d_angles = \
            core.compute_auto_jacobian_from_tensors(
                n, self.derivative_data.eye_pose_parameters.angles_deg
            )
        d_pupil_center_d_pose_and_dist = \
            torch.hstack((d_n_d_angles * dist, torch.eye(3), n.view(3, 1)))

        self.pupil_covariance_inUser = \
            self.compute_covariance_pupil_center_standalone(
                d_pupil_center_d_pose_and_dist,
                self.covariance,
                self.eye_shape_covariance
            )

        if node is not None:
            # Transform the covariance matrix to the coordinate system of the
            # node.

            # Covariances transform according to a rotation of the coordinate
            # system. The equation is driven by how covariance matrices
            # multiply the data. Let sup(A)x be a datum in the coordinate
            # system A, and let sup(B)Tsub(A) a transformation from the
            # coordinate A to B, so that sup(B)x = sup(B)Tsub(A). We use the
            # robotics notation of having the coordinate system as a
            # superscript on the left-hand side of the object. We further
            # simplify this by writing sup(A)x as A^x and sup(B)Tsub(A) as
            # B^T_A. Finally, quadratic forms in coordinate system A are
            # represented as A_S_A. We have:
            #
            # I = A^x.T * (A_S_A)^(-1) * (A^x)
            #
            # where I is an invariant (does not depend on coordinate system)
            # and S is the covariance matrix of x in the coordinate system A.
            # Then we have
            #
            # I = (A^T_B * B^x).T * (A_S_A)^(-1) * (A^T_B * B^x). = B^x.T *
            #   (A^T_B).T * (A_S_A)^(-1) * A^T_B * B^x = B^x.T * ((A^T_B)^(-1)
            #   * A_S_A * (A^T_B).T^(-1))^(-1) * B^x.
            #
            # Since (A^T_B)^(-1) = B^T_A, we have
            #
            # B_S_B = B^T_A * A_S_A * (B^T_A).T

            T_toNode_fromUser = \
                self.derivative_data.user.get_transform_toOther_fromSelf(node)
            R_toNode_fromUser = \
                T_toNode_fromUser.rotation.get_rotation_matrix()

            return \
                R_toNode_fromUser @ \
                self.pupil_covariance_inUser @ \
                R_toNode_fromUser.T
        else:
            return self.pupil_covariance_inUser

    def _compute_error_probability_for_pupil_center(
        self, coord, max_error_mm, min_error_mm=0.0, num_samples=100, node=None
    ):
        """Compute probabilities of error of pupil center in x, y, or z.

        This method computes the probabilities of error for x, y, or z
        components of the pupil center. An error is a deviation for the value
        of a given coordinate above a threshold. Thresholds are swept from a
        minimum to a maximum value in a given number of samples.

        The optional input "node" indicates the coordinate system in which the
        errors are computed. If node == None, the errors are computed in the
        coordinate system of the user.

        Args:
            coord (str): one of "x", "y", or "z", indicating which for which
            coordinate of eye position the probability of error should be
            computed.

            max_error_mm (float): maximum threshold value in mm.

            min_error_mm (float, optional): minimum threshold value in mm.
            Defaults to 0.0.

            num_samples (int, optional): number of thresholds from min to max.
            Defaults to 100.

            node (NodeModel, optional): secrets node object in which x, y, and z
            errors are expressed. Defaults to None, in which case the user
            coordinate system is used.
        """

        # Compute the cova
        pupil_covariance = self.compute_covariance_pupil_center(node=node)

        if coord == "x":
            sig = pupil_covariance[0, 0]
        elif coord == "y":
            sig = pupil_covariance[1, 1]
        else:
            sig = pupil_covariance[2, 2]

        thresholds_mm = torch.linspace(min_error_mm, max_error_mm, num_samples)
        normalized_thresholds_mm = \
            thresholds_mm / torch.sqrt(torch.tensor(2.0)) / sig

        return 1.0 - 1.0 * torch.special.erf(normalized_thresholds_mm)

    def _compute_d2_cost_glints_d2_eye_pose(self):
        """Compute 2nd deriv. of glint cost w.r.t. to eye pose.
        """

        d2_cost_glints_d2_param = 0

        for d_glint_d_param in self.d_all_glints_d_eye_pose:
            if d_glint_d_param is None:
                # Some glints are occluded.
                continue

            d2_cost_glints_d2_param = \
                d2_cost_glints_d2_param + d_glint_d_param.T @ d_glint_d_param

        return d2_cost_glints_d2_param

    def _compute_cross_derivative_limbus_terms(
        self,
        num_led_parameters,
        num_glint_measurements,
        num_visible_pupil_points
    ):
        """Compute contribution of limbus to cross derivative of cost function.

        The arguments provided are required to fill in the zero components of
        the derivative with blocks of the correct size.

        Args:
            num_led_parameters (int): number of visible leds.

            num_glint_measurements (int): 2 * number of visible glints.

            num_visible_pupil_points (int): number of visible pupil points.
        """
        d2_limbus_cost_d_eye_pose_d_extrinsics, \
            d2_limbus_cost_d_eye_pose_d_intrinsics, \
            d2_limbus_cost_d_eye_pose_d_limbus_measurements, \
            d2_limbus_cost_d_angles_d_extrinsics, \
            d2_limbus_cost_d_angles_d_intrinsics, \
            d2_limbus_cost_d_angles_d_limbus_measurements = \
            self._compute_d2_cost_x_d_optim_d_data("limbus")

        num_eye_pose = self.derivative_calculator_eye_pose.get_num_parameters()
        num_pupil_measurements = 2 * num_visible_pupil_points

        d2_F3_d_w2_d_z2 = torch.zeros(num_eye_pose, num_led_parameters)
        d2_F3_d_w2_d_z3 = d2_limbus_cost_d_eye_pose_d_extrinsics
        d2_F3_d_w2_d_z4 = d2_limbus_cost_d_eye_pose_d_intrinsics
        d2_F3_d_w2_d_z5 = torch.zeros(num_eye_pose, num_glint_measurements)
        d2_F3_d_w2_d_z6 = torch.zeros(num_eye_pose, num_pupil_measurements)

        d2_F3_d_w2_d_data = \
            torch.hstack(
                (
                    d2_F3_d_w2_d_z2,
                    d2_F3_d_w2_d_z3,
                    d2_F3_d_w2_d_z4,
                    d2_F3_d_w2_d_z5,
                    d2_F3_d_w2_d_z6
                )  # type: ignore
            )

        # All of the d_w3 terms are zero.
        num_data = d2_F3_d_w2_d_data.shape[1]
        d2_F3_d_w3_d_data = torch.zeros(num_visible_pupil_points, num_data)
        d2_F3_d_optim_d_data = \
            torch.vstack((d2_F3_d_w2_d_data, d2_F3_d_w3_d_data))

        # Now, we extend the matrix by adding the rows corresponding to
        # derivatives with respect to w4, i.e., limbus lifting parameters.
        num_limbus_lifting_angles = \
            self.derivative_calculator_limbus_lifting.get_num_parameters()
        d2_F3_d_w4_d_z2 = \
            torch.zeros(num_limbus_lifting_angles, num_led_parameters)
        d2_F3_d_w4_d_z3 = d2_limbus_cost_d_angles_d_extrinsics
        d2_F3_d_w4_d_z4 = d2_limbus_cost_d_angles_d_intrinsics

        d2_F3_d_w4_d_z5 = \
            torch.zeros(num_limbus_lifting_angles, num_glint_measurements)
        d2_F3_d_w4_d_z6 = \
            torch.zeros(num_limbus_lifting_angles, num_pupil_measurements)

        d2_F3_d_w4_d_data = \
            torch.hstack(
                (
                    d2_F3_d_w4_d_z2,
                    d2_F3_d_w4_d_z3,
                    d2_F3_d_w4_d_z4,
                    d2_F3_d_w4_d_z5,
                    d2_F3_d_w4_d_z6,
                )  # type: ignore
            )

        # Extend the matrix by adding the columns corresponding to derivatives
        # with respect to z7, i.e., limbus measurements.
        d2_F_d_w2_d_z7 = d2_limbus_cost_d_eye_pose_d_limbus_measurements
        num_limbus_measurements = d2_F_d_w2_d_z7.shape[1]
        d2_F_d_w3_d_z7 = \
            torch.zeros(num_visible_pupil_points, num_limbus_measurements)
        d2_F_d_w4_d_z7 = d2_limbus_cost_d_angles_d_limbus_measurements
        d2_F3_d_optim_d_z7 = \
            torch.vstack((d2_F_d_w2_d_z7, d2_F_d_w3_d_z7, d2_F_d_w4_d_z7))

        return d2_F3_d_optim_d_data, d2_F3_d_w4_d_data, d2_F3_d_optim_d_z7

    def _compute_cross_derivative_eye_shape(self):
        """Compute cross derivative of cost with eye shape as data term.
        """

        # w = (w2 = eye pose, w3 = pupil, w4 = limbus (opt.))
        #
        # z = (z1 = eye shape, z2, ..., z6, z7 (opt.))
        #
        # The terms "w3 = pupil" and "w4 = limbus" (which is optional) refer to
        # the lifting parameters required to locate points along the pupil and
        # limbus contours.

        d2_cost_glints_d_eye_pose_d_eye_shape = \
            self._compute_d2_cost_glints_d_optim_d_eye_shape()
        d2_cost_pupil_d_eye_pose_d_eye_shape, \
            d2_cost_pupil_d_angles_d_eye_shape = \
            self._compute_d2_cost_x_d_optim_d_eye_shape("pupil")

        d2_F_d_w2_d_z1 = \
            d2_cost_glints_d_eye_pose_d_eye_shape + \
            d2_cost_pupil_d_eye_pose_d_eye_shape
        d2_F_d_w3_d_z1 = d2_cost_pupil_d_angles_d_eye_shape

        if self.with_limbus:
            d2_cost_limbus_d_eye_pose_d_eye_shape, \
                d2_cost_limbus_d_angles_d_eye_shape = \
                self._compute_d2_cost_x_d_optim_d_eye_shape("limbus")
            d2_F_d_w2_d_z1 = \
                d2_F_d_w2_d_z1 + d2_cost_limbus_d_eye_pose_d_eye_shape
            d2_F_d_w4_d_z1 = d2_cost_limbus_d_angles_d_eye_shape
        else:
            d2_F_d_w4_d_z1 = \
                torch.zeros(
                    0,
                    self.derivative_calculator_eye_shape.get_num_parameters()
                )

        return \
            torch.vstack(
                (d2_F_d_w2_d_z1, d2_F_d_w3_d_z1, d2_F_d_w4_d_z1)  # type:ignore
            )

    def _compute_d2_cost_x_d2_optimization(self, feature_name):
        """Compute 2nd deriv. of pupil/limbus cost w.r.t. optimization params.

        In the method's name and implementation "x" is a stand in for "pupil"
        or "limbus".

        Args:
            feature_name (str): One of "pupil" or "limbus" depending on which
            component of the cost function is considered.
        """

        if feature_name.lower() == "pupil":
            d_all_x_d_eye_pose = self.d_pupil_d_eye_pose
            d_all_x_d_angles = self.d_pupil_d_angles
        else:
            d_all_x_d_eye_pose = self.d_limbus_d_eye_pose
            d_all_x_d_angles = self.d_limbus_d_angles

        d2_cost_x_d2_eye_pose = 0
        d2_cost_x_d2_angles = 0
        d2_cost_x_d_eye_pose_d_angles = 0

        for i in range(len(d_all_x_d_eye_pose)):
            d_x_d_eye_pose = d_all_x_d_eye_pose[i]
            d_x_d_angles = d_all_x_d_angles[i]
            if d_x_d_eye_pose is None or d_x_d_angles is None:
                # Some pupil points are not visible.
                continue

            # But they are visible, both derivatives (w.r.t eye pose and w.r.t.
            # angles) will be valid.
            d2_cost_x_d2_eye_pose = \
                d2_cost_x_d2_eye_pose + d_x_d_eye_pose.T @ d_x_d_eye_pose

            d2_cost_x_d2_angles = \
                d2_cost_x_d2_angles + d_x_d_angles.T @ d_x_d_angles

            d2_cost_x_d_eye_pose_d_angles = \
                d2_cost_x_d_eye_pose_d_angles + \
                d_x_d_eye_pose.T @ d_x_d_angles

        return \
            d2_cost_x_d2_eye_pose, \
            d2_cost_x_d2_angles, \
            d2_cost_x_d_eye_pose_d_angles

    # This and the next method compute components of the second derivative of
    # the cost function with respect to optimization and data parameters.
    def _compute_d2_cost_glints_d_optim_d_data(self, optim_param):
        """Compute cross deriv. of glint errors w.r.t. optim. and data params.

        Note: We treat the derivative with respect to eye shape in a special
        way and omit it from the current computation, because eye shape can be
        both an optimization parameter (during user calibration) and a data
        parameter (during eye tracking).
        """

        d2_cost_glints_d_param_d_leds = 0
        d2_cost_glints_d_param_d_extrinsics = 0
        d2_cost_glints_d_param_d_intrinsics = 0

        # We better use an index than try to zip all the parameters.
        for i in range(len(optim_param)):
            d_glint_d_param = optim_param[i]
            if d_glint_d_param is None:
                continue

            d_glint_d_leds = self.d_all_glints_d_leds[i]
            d_glint_d_extrinsics = self.d_all_glints_d_camera_extrinsics[i]
            d_glint_d_intrinsics = self.d_all_glints_d_camera_intrinsics[i]

            d2_cost_glints_d_param_d_leds = \
                d2_cost_glints_d_param_d_leds + \
                d_glint_d_param.T @ d_glint_d_leds

            d2_cost_glints_d_param_d_extrinsics = \
                d2_cost_glints_d_param_d_extrinsics + \
                d_glint_d_param.T @ d_glint_d_extrinsics

            d2_cost_glints_d_param_d_intrinsics = \
                d2_cost_glints_d_param_d_intrinsics + \
                d_glint_d_param.T @ d_glint_d_intrinsics

        # A special parameter. This is the derivative with respect to glint
        # measurements of the transpose of the derivative of the glint error
        # with respect to eye pose. This derivative has size (# eye pose
        # parameters) x (2 x # visible glints), where the factor two comes from
        # the fact that glints are two dimensional.
        d_all_visible_glints_d_param = \
            [d for d in optim_param if d is not None]
        # Note the negative sign.
        d2_cost_glints_d_param_d_glint_measurements = \
            -1 * torch.vstack(d_all_visible_glints_d_param).T

        return \
            d2_cost_glints_d_param_d_leds, \
            d2_cost_glints_d_param_d_extrinsics, \
            d2_cost_glints_d_param_d_intrinsics, \
            d2_cost_glints_d_param_d_glint_measurements

    def _compute_d2_cost_x_d_optim_d_data(self, feature_name):
        """Compute cross deriv. of pupil/limbus errors w.r.t. optim and data.

        In the method's name and implementation "x" is a stand in for "pupil"
        or "limbus".

        Args:
            feature_name (str): One of "pupil" or "limbus" depending on which
            component of the cost is considered.
        """

        if feature_name.lower() == "pupil":
            # Optimization parameters
            d_all_x_d_eye_pose = self.d_pupil_d_eye_pose
            d_all_x_d_angles = self.d_pupil_d_angles

            # Data
            d_all_x_d_extrinsics = self.d_pupil_d_camera_extrinsics
            d_all_x_d_intrinsics = self.d_pupil_d_camera_intrinsics
        else:
            # Optimization parameters
            d_all_x_d_eye_pose = self.d_limbus_d_eye_pose
            d_all_x_d_angles = self.d_limbus_d_angles

            # Data
            d_all_x_d_extrinsics = self.d_limbus_d_camera_extrinsics
            d_all_x_d_intrinsics = self.d_limbus_d_camera_intrinsics

        d2_cost_x_d_eye_pose_d_extrinsics = 0
        d2_cost_x_d_eye_pose_d_intrinsics = 0
        d2_cost_x_d_angles_d_extrinsics = 0
        d2_cost_x_d_angles_d_intrinsics = 0

        # Better to use an index than try to zip all these parameters.
        for i in range(len(d_all_x_d_eye_pose)):
            d_x_d_eye_pose = d_all_x_d_eye_pose[i]
            d_x_d_angles = d_all_x_d_angles[i]
            if d_x_d_eye_pose is None or d_x_d_angles is None:
                continue

            d_x_d_extrinsics = d_all_x_d_extrinsics[i]
            d_x_d_intrinsics = d_all_x_d_intrinsics[i]

            d2_cost_x_d_eye_pose_d_extrinsics = \
                d2_cost_x_d_eye_pose_d_extrinsics + \
                d_x_d_eye_pose.T @ d_x_d_extrinsics

            d2_cost_x_d_eye_pose_d_intrinsics = \
                d2_cost_x_d_eye_pose_d_intrinsics + \
                d_x_d_eye_pose.T @ d_x_d_intrinsics

            d2_cost_x_d_angles_d_extrinsics = \
                d2_cost_x_d_angles_d_extrinsics + \
                d_x_d_angles.T @ d_x_d_extrinsics

            d2_cost_x_d_angles_d_intrinsics = \
                d2_cost_x_d_angles_d_intrinsics + \
                d_x_d_angles.T @ d_x_d_intrinsics

        # Special parameters. These are the derivative with respect to
        # pupil/limbus measurements of the transpose of the derivative of the
        # pupil/limbus error with respect to eye pose or lifting parameter.
        # This derivative has size (dim optimization parameter) x (2 x #
        # visible pupil/limbus points), where the factor two comes from the
        # fact that pupil/limbus points are two dimensional.
        d_all_visible_x_d_eye_pose = \
            [d for d in d_all_x_d_eye_pose if d is not None]
        d2_cost_x_d_eye_pose_d_x_measurements = \
            -1 * torch.vstack(d_all_visible_x_d_eye_pose).T

        d_all_visible_x_d_angles = \
            [d for d in d_all_x_d_angles if d is not None]
        d2_cost_x_d_angles_d_x_measurements = \
            -1 * torch.vstack(d_all_visible_x_d_angles).T

        return \
            d2_cost_x_d_eye_pose_d_extrinsics, \
            d2_cost_x_d_eye_pose_d_intrinsics, \
            d2_cost_x_d_eye_pose_d_x_measurements, \
            d2_cost_x_d_angles_d_extrinsics, \
            d2_cost_x_d_angles_d_intrinsics, \
            d2_cost_x_d_angles_d_x_measurements

    # This method computes the second derivative of the glint cost with respect
    # to optimization parameters and eye shape. This is done separately because
    # eye shape may play the role of an optimization parameter during user
    # calibration.
    def _compute_d2_cost_glints_d_optim_d_eye_shape(self):
        """Compute the second derivative of the glint cost with respect to
        optimization parameters and eye shape.
        """

        d2_cost_glints_d_eye_pose_d_eye_shape = 0

        # Better to use an index than try to zip all these parameters.
        for i in range(len(self.d_all_glints_d_eye_pose)):
            d_glint_d_eye_pose = self.d_all_glints_d_eye_pose[i]
            if d_glint_d_eye_pose is None:
                # Some glints are occluded.
                continue

            d_glint_d_eye_shape = self.d_all_glints_d_eye_shape[i]

            d2_cost_glints_d_eye_pose_d_eye_shape = \
                d2_cost_glints_d_eye_pose_d_eye_shape + \
                d_glint_d_eye_pose.T @ d_glint_d_eye_shape

        return d2_cost_glints_d_eye_pose_d_eye_shape

    # This method computes the second derivative of the pupil/limbus cost with
    # respect to optimization parameters and eye shape. Also done separately.
    def _compute_d2_cost_x_d_optim_d_eye_shape(self, feature_name):
        """Compute cross deriv. of pupil/limbus errors w.r.t. optim and shape.

        In the method's name and implementation "x" is a stand in for "pupil"
        or "limbus".

        Args:
            feature_name (str): One of "pupil" or "limbus" depending on which
            component of the cost is considered.
        """

        if feature_name.lower() == "pupil":
            d_all_x_d_eye_pose = self.d_pupil_d_eye_pose
            d_all_x_d_angles = self.d_pupil_d_angles
            d_all_x_d_eye_shape = self.d_pupil_d_eye_shape
        else:
            d_all_x_d_eye_pose = self.d_limbus_d_eye_pose
            d_all_x_d_angles = self.d_limbus_d_angles
            d_all_x_d_eye_shape = self.d_limbus_d_eye_shape

        d2_cost_x_d_eye_pose_d_eye_shape = 0
        d2_cost_x_d_angles_d_eye_shape = 0

        for i in range(len(d_all_x_d_angles)):
            d_x_d_eye_pose = d_all_x_d_eye_pose[i]
            d_x_d_angles = d_all_x_d_angles[i]
            if d_x_d_eye_pose is None or d_x_d_angles is None:
                continue

            d_x_d_eye_shape = d_all_x_d_eye_shape[i]

            d2_cost_x_d_eye_pose_d_eye_shape = \
                d2_cost_x_d_eye_pose_d_eye_shape + \
                d_x_d_eye_pose.T @ d_x_d_eye_shape

            d2_cost_x_d_angles_d_eye_shape = \
                d2_cost_x_d_angles_d_eye_shape + \
                d_x_d_angles.T @ d_x_d_eye_shape

        return d2_cost_x_d_eye_pose_d_eye_shape, d2_cost_x_d_angles_d_eye_shape
