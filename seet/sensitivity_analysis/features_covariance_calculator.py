"""Class for computing contrib. of feat. error to eye-pose and -shape error.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.sensitivity_analysis import \
    base_covariance_calculator, \
    sensitivity_analysis_configs
from seet.sensitivity_analysis.sensitivity_analysis_utils import \
    build_cross_cov_matrix, \
    build_single_cov_matrix
import os
import torch


class FeaturesCovarianceCalculator(
    base_covariance_calculator.CovarianceCalculator
):
    """Class for computing covariance of image feats. for sensitivity analysis.
    """

    def __init__(
        self,
        parameter_file_name=os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            r"default_covariances/default_feature_covariances.json"
        )
    ):
        """Initialize covariances of image features (glint, pupil, and limbus).

        Args:
            parameter_file_name (str, optional): name of parameter file with
            configuration of sensitivity parameters of input features. Defaults
            to os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            "default_covariances/default_feature_covariances.json" ).
        """
        super().__init__(parameter_file_name=parameter_file_name)

        self.baseline_stds = torch.empty(0)
        member_names = ("glint", "pupil point", "limbus point")
        for name in member_names:
            feature_key = name + "s"
            feature_dict = self.parameters[feature_key]
            std = torch.tensor(feature_dict["std"])
            self.baseline_stds = \
                torch.hstack((self.baseline_stds, std))
            cor = torch.tensor(feature_dict["intra cor"])

            # Replace empty spaces in the name with a subscript.
            name = name.replace(" ", "_")

            # Creates member variable "x_single_cov", where x is "glint",
            # "pupil", or "limbus".
            member_single_cov = name + "_single_cov"
            self.__dict__[member_single_cov] = \
                build_single_cov_matrix(std, cor)

            # Creates member variable "x_inter_cor", where x is "glint",
            # "pupil", or "limbus".
            member_inter_cor = name + "_inter_cor"
            self.__dict__[member_inter_cor] = feature_dict.get("inter cor", 0)

            # Creates member variable "x_cross_cov", where x is "glint",
            # "pupil", or "limbus".
            member_cross_cov = name + "_cross_cov"
            self.__dict__[member_cross_cov] = \
                build_cross_cov_matrix(
                    self.__dict__[member_single_cov],
                    inter_cor=self.__dict__[member_inter_cor]
            )

        self.weights = torch.ones_like(self.baseline_stds)

        # If the baseline stds are zero, we switch the role of the baseline
        # stds and the weights.
        self._switch_zero_std_with_weight()

    def _compute_individual_contribution(
        self,
        feature_type,
        d_eye_params_d_input_params,
        d_eye_params_d_input_params_indices,
        with_limbus
    ):
        """Compute contrib. of glint, pupil, or limbus feat. to eye-param cov.

        d_eye_param_d_input_params is a (M, N) torch tensor corresponding to
        the derivative of eye-shape (M = 0) or eye-pose (M = 5) parameters with
        respect to all input parameters. For eye-pose parameters, the input
        parameters start with N0 eye-shape parameters (N0 = 9). For eye-shape
        parameters these N0 parameters are absent. In either case the remaining
        parameters are N1 LED parameters (N1 = 3 * M1, where M1 is the number
        of LEDs), N2 = 6 camera extrinsic parameters, N3 = 9 camera intrinsic
        parameters, and N4_k glint parameters (N4_k = 2 * M4_k, where M4_k is
        the number of glints detected in image frame k), N5_k pupil-point
        parameters (N5_k = 2 * M5_k, where M5_k is the number of pupil points
        detected in image frame k), and, optionally, N6_k limbus parameters
        (N6_k = 2 * M6_k, where M6_k is the number of limbus points detected in
        image frame k).

        d_eye_param_d_input_params_indices is a list with the beginning and
        final plus 1 columns to which each of the image-feature input
        parameters correspond in d_eye_param_d_all_parameters. For example, if
        eye param refers to eye pose and was computed using 9 eye-shape
        parameters, 12 LEDs and images captured across two frames, such that in
        the first frame 12 glints, 25 pupil points, and 27 limbus points were
        detected, and in the second frame 11 glints, 28 pupil points, and 23
        limbus points were detected, L = d_shape_d_all_parameter_indices will
        be

        L = [60, 84, 129, 181, 205, 261, 307], where

        L[0] is the first column of the first column of the glints block in
        d_eye_params_d_input_params for frame 1, L[1] is the first column of
        the pupil block for frame 1, L[2] is the first column of the limbus
        block for frame 1, L[3] is the first column of the glints block for
        frame 2, L[4] is the first column of the pupil block for frame 2, L[5]
        is the first column of the limbus block for frame 2.

        To extract the pupil block for frame 1, for example, we'd index
        d_shape_d_all_parameters as

        D_pupil_frame_1 = d_shape_d_all_parameters[:, L[0]:L[1]]

        Args:
            feature_type (string): one of "glint", "pupil", or "limbus".

            d_eye_params_d_input_params (torch.Tensor): (M, N) torch tensor
            corresponding to the derivative of eye-pose or eye-shape parameters
            with respect to all input parameters in K image frames.

            d_eye_params_d_input_params_indices (list of int): list with
            positional indices of all input parameters as columns of
            d_eye_params_d_input_params.
        """
        # LEDs (0), cam. extr. (1), cam. intr. (2), glints (3), pupil (4)
        if feature_type == "glint":
            start = 0
            single_cov = self.glint_single_cov
            inter_cor = self.glint_inter_cor
            cross_cov = self.glint_cross_cov
        elif feature_type == "pupil":
            start = 1
            single_cov = self.pupil_point_single_cov
            inter_cor = self.pupil_point_inter_cor
            cross_cov = self.pupil_point_cross_cov
        else:  # assumes feature_type == "limbus" is true.
            # If feature_type is limbus and with_limbus is false, there are no
            # limbus points.
            if with_limbus is False:
                num_eye_params = d_eye_params_d_input_params.shape[0]
                return torch.zeros((num_eye_params, num_eye_params))

            start = 2
            single_cov = self.limbus_point_single_cov
            inter_cor = self.limbus_point_inter_cor
            cross_cov = self.limbus_point_cross_cov

        step = 2  # glints (+1), pupil (+1)
        if with_limbus:
            step = step + 1

        C = 0
        dim = 2  # All features are two-dimensional.
        W = torch.eye(dim) * self.weights[start]
        stop = len(d_eye_params_d_input_params_indices)

        # Each feature type corresponds to a block of columns of the
        # derivative of the eye shape. We subtract by 1 because the stopping
        # point of the last block is not the end point of a new block.
        for block_idx in range(start, stop - 1, step):
            feat_start = d_eye_params_d_input_params_indices[block_idx]
            # The addition by 1 below works because we subtracted stop by 1.
            feat_stop = d_eye_params_d_input_params_indices[block_idx + 1]

            # Each block includes the contribution of multiple features of the
            # same type.
            for idx_i in range(feat_start, feat_stop, dim):
                D_i = d_eye_params_d_input_params[:, idx_i:(idx_i + dim)]
                D_i = D_i @ W
                tmp = D_i @ single_cov @ D_i.T
                C = C + tmp

                # Include contribution from cross correlation, if any.
                if inter_cor != 0.0:
                    start_ = idx_i + dim
                    for idx_j in range(start_, feat_stop, dim):
                        D_j = \
                            d_eye_params_d_input_params[:, idx_j:(idx_j + dim)]
                        D_j = D_j @  W
                        tmp = D_i @ cross_cov @ D_j.T
                        C = C + tmp + tmp.T

        return C

    def compute_covariance(
        self,
        param_type,
        d_eye_params_d_input_params,
        d_eye_params_d_input_params_indices,
        with_limbus
    ):
        """Compute contribution of feature noise to the cov. of eye params.

        Args:
            param_type (str): one of "pose" or "shape".

            d_eye_params_d_input_params (torch.Tensor): (M, N) torch tensor
            corresponding to the derivative of eye-shape or -pose parameters
            with respect to all input parameters in K image frames.

            d_eye_params_d_input_params_indices (list of int): list with
            positional indices of all input parameters as columns of
            d_eye_params_d_input_params.

            with_limbus (bool): whether the derivative of the eye parameters
            include terms for the limbus.
        """

        super().compute_covariance(
            param_type,
            d_eye_params_d_input_params,
            d_eye_params_d_input_params_indices
        )

        if param_type == "pose":
            start = 4  # shape (0), LEDs (1), cam. extr. (2), cam. intr. (3)
        else:  # assumes that param_type == "shape" is True
            start = 3  # LEDs (0), cam. extr. (1), cam. intr. (2)

        feature_types = ("glint", "pupil")
        if with_limbus:
            feature_types = feature_types + ("limbus",)

        C = 0
        for feature in feature_types:
            C_feat = \
                self._compute_individual_contribution(
                    feature,
                    d_eye_params_d_input_params[:self.num_eye_params, :],
                    d_eye_params_d_input_params_indices[start:],
                    with_limbus
                )
            C = C + C_feat

        return C
