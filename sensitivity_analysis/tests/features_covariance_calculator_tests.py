"""Test contribution of image features to eye-shape and -pose covariances.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import sensitivity_analysis as sensitivity_analysis
from sensitivity_analysis.tests import test_utils
from parameterized import parameterized_class
import torch


@parameterized_class(
    ("with_limbus", "subsystem_idx"),
    [
        (with_limbus, subsystem_idx)
        for with_limbus in (True, False)
        for subsystem_idx in (0, 1)
    ]
)
class TestFeaturesCovarianceCalculator(test_utils.TestCommonUtils):
    """Unit tests for sensitivity-analysis utilities.
    """

    def setUp(self):
        """Initialize data for tests
        """

        super().setUp()

        self.derivative_data.set_eye_and_subsystem(index=self.subsystem_idx)
        self.shape_cov_calc = \
            sensitivity_analysis.EyeShapeCovariance(self.derivative_data)
        self.shape_cov_calc.with_limbus = self.with_limbus

    def test_init(self):
        """Test default initialization of FeaturesInputCovariance object.
        """

        feature_cov_calculator = \
            sensitivity_analysis.FeaturesCovarianceCalculator()
        self.assertTrue(
            torch.allclose(
                feature_cov_calculator.glint_single_cov,
                torch.diag(torch.tensor([0.25, 0.25])**2)
            )
        )
        self.assertTrue(feature_cov_calculator.glint_inter_cor == 0.0)
        self.assertTrue(
            torch.allclose(
                feature_cov_calculator.glint_cross_cov,
                torch.zeros((2, 2))
            )
        )

    def test_compute_covariance(self):
        """Test computation of contribution of features to shape covariance.
        """

        # Generate data.
        list_gaze_rotation_deg = \
            [torch.tensor([15.0, -10.0]), torch.tensor([-20.0, 15.0])]
        self.shape_cov_calc.set_list_of_gaze_angles(list_gaze_rotation_deg)
        d_eye_param_d_data = \
            self.shape_cov_calc.compute_d_optim_d_data()
        d_eye_param_d_data_indices = \
            self.shape_cov_calc.compute_d_optim_d_data_indices()

        feature_cov_calculator = \
            sensitivity_analysis.FeaturesCovarianceCalculator()

        # Each feature separately.
        start = 3  # LEDs (0), cam. extr. (1), cam. intr. (2)
        num_eye_params = \
            sensitivity_analysis.EyeShapeDerivatives.get_num_parameters()
        C = 0
        for feature_type in ("glint", "pupil", "limbus"):
            C_feat = \
                feature_cov_calculator._compute_individual_contribution(
                    feature_type,
                    d_eye_param_d_data[:num_eye_params, :],
                    d_eye_param_d_data_indices[start:],
                    self.shape_cov_calc.with_limbus
                )

            self.assertTrue(sensitivity_analysis.is_valid_covariance(C_feat))

            C = C + C_feat

        # All together.
        C_ = \
            feature_cov_calculator.compute_covariance(
                "shape",
                d_eye_param_d_data,
                d_eye_param_d_data_indices,
                self.shape_cov_calc.with_limbus
            )

        self.assertTrue(torch.allclose(C, C_))
