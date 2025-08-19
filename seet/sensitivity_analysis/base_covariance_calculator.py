"""Base class for computing contr. of input to eye-pose and -shape error.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import seet.sensitivity_analysis.derivative_calculators as \
    derivative_calculators
import torch


class CovarianceCalculator():
    """Base class for contribution of input to covariance of output.
    """

    def __init__(
        self,
        parameter_file_name=None
    ):
        """Initialize common components.

        Args:
            parameter_file_name (str, optional): name of json configuration
            file.
        """

        self.parameter_file_name = parameter_file_name
        with open(self.parameter_file_name) as parameter_file_stream:
            self.parameters = json.load(parameter_file_stream)

        # Multiplicative weight applied to the covariance.
        self.baseline_stds = 1.0
        self.weights = 1.0

    def _switch_zero_std_with_weight(self):
        """Switch stds and weight values if stds are zero.
        """

        for i in range(len(self.baseline_stds)):
            if self.baseline_stds[i] == 0.0:
                self.baseline_stds[i] = self.weights[i]
                self.weights[i] = 0
                self.cov[i, i] = self.baseline_stds[i] ** 2

    def set_weights(self, weights):
        """Apply weights to the standard deviations.

        Args:
            weights (float): weights to be applied to the standard deviations.

            component (str): parameter whose weight is to be set.
        """

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        self.weights = weights

    def set_stds(self, stds):
        """Reset the values of the standard deviations.

        Since a weight is always applied to the covariance matrices, the change
        is stds can be accomplished by changing the weight.

        Args:
            stds (torch.Tensor): new value of standard deviations.
        """

        if isinstance(self.baseline_stds, torch.Tensor):
            if isinstance(stds, float):
                stds = stds * torch.ones_like(self.baseline_stds)
            elif not isinstance(stds, torch.Tensor):
                stds = torch.tensor(stds)

        # What if the self.baseline_stds is zero?
        self.set_weights(stds / self.baseline_stds)

    def compute_covariance(
        self,
        param_type,
        d_eye_params_d_input_params,
        d_eye_params_d_input_params_indices
    ):
        """Compute common elements of contribution of input to noise.

        Args:
            param_type (str): one of "pose" or "shape".

            d_eye_params_d_input_params (torch.Tensor): (M, N) torch tensor
            corresponding to the derivative of eye-shape or -pose parameters
            with respect to all input parameters in K image frames.

            d_eye_params_d_input_params_indices (list of int): list with
            positional indices of all input parameters as columns of
            d_eye_params_d_input_params.
        """

        if param_type == "pose":
            self.num_eye_params = \
                derivative_calculators.EyePoseDerivatives.get_num_parameters()
        else:  # Assumes that param_type == "shape" is True.
            self.num_eye_params = \
                derivative_calculators.EyeShapeDerivatives.get_num_parameters()
