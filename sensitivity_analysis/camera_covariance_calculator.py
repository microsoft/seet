"""Class for computing covariance of camera parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from sensitivity_analysis import \
    base_covariance_calculator, \
    sensitivity_analysis_configs
from sensitivity_analysis.sensitivity_analysis_utils import \
    build_single_cov_matrix, \
    stack_covariances
import os
import torch


class CameraCovarianceCalculator(
    base_covariance_calculator.CovarianceCalculator
):
    """Class for computing covariance of camera extr. for sensitivity analysis.
    """

    @staticmethod
    def _extract_cov(input_dict, key, key_):
        """Extract covariance matrix of params from dictionary using keys.

        Args:
            intrinsics_dict (dict): dictionary holding parameters.

            key, key_ (str): keys holding specific parameters.

        Returns:
            torch.Tensor: covariance matrix.
        """

        cov = input_dict.get("cov")
        if cov is None:
            first_param_dict = input_dict[key]
            std = torch.tensor(first_param_dict["std"])
            cor = torch.tensor(first_param_dict["cor"])
            first_cov = build_single_cov_matrix(std, cor)

            second_param_dict = input_dict[key_]
            std = torch.tensor(second_param_dict["std"])
            cor = torch.tensor(second_param_dict["cor"])
            second_cov = build_single_cov_matrix(std, cor)

            cov = stack_covariances(first_cov, second_cov)

        return cov

    def __init__(
        self,
        parameter_file_name=os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            r"default_covariances/default_camera_covariances.json"
        )
    ):
        """Initialize class object using configuration file

        Args:
            parameter_file_name (str, optional): configuration file. Defaults
            to os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            "default_covariances/default_camera_covariances.json" ).
        """

        super().__init__(parameter_file_name=parameter_file_name)

        # Get the full covariance matrix, if available.
        self.cov = self.parameters.get("cov")

        if self.cov is None:
            # Create covariance for extrinsic parameters.
            # Full covariance matrix is not available. Get individual
            # covariance and cross-covariance components.
            extrinsic_parameters = self.parameters["extrinsics"]
            self.extrinsics_cov = \
                CameraCovarianceCalculator._extract_cov(
                    extrinsic_parameters, "translation", "rotation"
                )

            intrinsic_parameters = self.parameters["intrinsics"]
            self.intrinsics_cov = intrinsic_parameters.get("cov")
            if self.intrinsics_cov is None:
                self.pinhole_cov = \
                    CameraCovarianceCalculator._extract_cov(
                        intrinsic_parameters["pinhole"],
                        "focal length",
                        "principal point"
                    )

                self.distortion_cov = \
                    CameraCovarianceCalculator._extract_cov(
                        intrinsic_parameters["distortion"],
                        "distortion center",
                        "distortion coefficients"
                    )

                self.intrinsics_cov = \
                    stack_covariances(self.pinhole_cov, self.distortion_cov)

            self.cov = \
                stack_covariances(self.extrinsics_cov, self.intrinsics_cov)

        self.baseline_stds = torch.sqrt(torch.diag(self.cov))
        self.weights = torch.ones_like(self.baseline_stds)

        # If the baseline stds are zero, we switch the role of the baseline
        # stds and the weights.
        self._switch_zero_std_with_weight()

    def compute_covariance(
        self,
        param_type,
        d_eye_params_d_input_params,
        d_eye_params_d_input_params_indices
    ):
        """Compute contribution of LED noise to covariance of eye parameters.

        Args:
            param_type (str): one of "pose" or "shape".

            d_eye_params_d_input_params (torch.Tensor): (M, N) torch tensor
            corresponding to the derivative of eye parameters with respect to
            all input parameters in K image frames.

            d_eye_params_d_input_params_indices (list of int): list with
            positional indices of all input parameters as columns of
            d_shape_d_input_params.
        """

        super().compute_covariance(
            param_type,
            d_eye_params_d_input_params,
            d_eye_params_d_input_params_indices
        )

        if param_type == "pose":
            block_idx = 2  # shape (0), LED (1), cam. extr. (this, 2), etc.
        else:  # Assumes that param_type == "shape" is True.
            block_idx = 1  # LED (this, 0), cam. extr. (this, 1), etc.

        # Starting point for extrinsic parameters.
        feat_start = d_eye_params_d_input_params_indices[block_idx]

        # The next block, indexed by block_idx + 1, is the starting point for
        # intrinsic parameters. We end at the end of that block, so we add 2.
        feat_stop = d_eye_params_d_input_params_indices[block_idx + 2]

        D = \
            d_eye_params_d_input_params[
                :self.num_eye_params, feat_start:feat_stop
            ]

        D = D @ torch.diag(self.weights)
        return D @ self.cov @ D.T
