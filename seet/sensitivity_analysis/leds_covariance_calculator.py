"""Class for computing contrib. of calib error to eye-pose and -shape error.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.sensitivity_analysis import \
    base_covariance_calculator, \
    sensitivity_analysis_configs, \
    sensitivity_analysis_utils
from seet.sensitivity_analysis.sensitivity_analysis_utils import \
    build_cross_cov_matrix, \
    build_single_cov_matrix
import os
import torch


class LEDsCovarianceCalculator(
    base_covariance_calculator.CovarianceCalculator
):
    """Class for computing covariance of LEDs for sensitivity analysis.
    """

    def __init__(
        self,
        parameter_file_name=os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            r"default_covariances/default_led_covariances.json"
        )
    ):
        """Initialize covariances of image features (glint, pupil, and limbus).

        Args:
            parameter_file_name (str, optional): name of parameter file with
            configuration of sensitivity parameters of input features. Defaults
            to os.path.join(
            sensitivity_analysis_configs.SENSITIVITY_ANALYSIS_DIR,
            "default_covariances/default_led_covariances.json" ).
        """

        super().__init__(parameter_file_name=parameter_file_name)

        # Get the full covariance matrix, if available.
        self.cov = self.parameters.get("cov")

        if self.cov is None:
            # Full covariance matrix is not available. Get individual
            # covariance and cross-covariance components.
            std = torch.tensor(self.parameters["std"])
            cor = torch.tensor(self.parameters["intra cor"])
            self.single_cov = build_single_cov_matrix(std, cor)
            self.inter_cor = self.parameters.get("inter cor", 0)
            self.cross_cov = \
                build_cross_cov_matrix(self.single_cov, self.inter_cor)
            self.baseline_stds = std
        else:
            self.baseline_stds = torch.sqrt(torch.diag(self.cov[:3, :3]))

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

            d_eye_params_d_input_params (torch.Tensor): (9, N) torch tensor
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
            block_idx = 1  # shape (0), LED (this, 1), cam. extr. (2), etc.
        else:  # Assumes that param_type == "shape" is True.
            block_idx = 0  # LED (this, 0), cam. extr. (1), etc.

        feat_start = d_eye_params_d_input_params_indices[block_idx]
        # The addition by 1 below works because we subtracted stop by 1.
        feat_stop = d_eye_params_d_input_params_indices[block_idx + 1]

        D = d_eye_params_d_input_params[:self.num_eye_params, :]
        W = torch.diag(self.weights)

        if self.cov is not None:
            W_ = torch.empty((0, 0))
            for led_idx in range(feat_start, feat_stop, 3):  # LEDs are 3D
                W_ = sensitivity_analysis_utils.stack_covariances(W_, W)

            # Full computation through a single matrix multiplication.
            D_ = D[:, feat_start:feat_stop] @ W_
            return D_ @ self.cov @ D_.T

        else:
            # Each block includes the contribution of multiple features of the
            # same type.
            C = 0
            dim = 3  # LEDs positions are three dimensional.
            for idx_i in range(feat_start, feat_stop, dim):
                D_i = D[:, idx_i:(idx_i + dim)] @ W
                tmp = D_i @ self.single_cov @ D_i.T
                C = C + tmp

                # Include contribution from cross correlation, if any.
                if self.inter_cor != 0.0:
                    start_ = idx_i + dim
                    for idx_j in range(start_, feat_stop, dim):
                        D_j = D[:, idx_j:(idx_j + dim)] @ W
                        tmp = D_i @ self.cross_cov @ D_j.T
                        C = C + tmp + tmp.T

            return C
