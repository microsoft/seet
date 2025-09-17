"""Test contribution of leds to shape and pose covariances.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import sensitivity_analysis as sensitivity_analysis
from tests.sensitivity_analysis_tests import test_utils
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
class TestLEDsCovarianceCalculator(test_utils.TestCommonUtils):
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
        """Test default initialization of LEDsInputCovariance object.
        """

        led_cov_calculator = \
            sensitivity_analysis.LEDsCovarianceCalculator()
        self.assertTrue(
            torch.allclose(
                led_cov_calculator.single_cov,
                torch.diag(torch.tensor([0.5, 0.5, 0.5])**2)
            )
        )
        self.assertTrue(led_cov_calculator.inter_cor == 0.0)
        self.assertTrue(
            torch.allclose(
                led_cov_calculator.cross_cov,
                torch.zeros((3, 3))
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

        led_cov_calculator = \
            sensitivity_analysis.LEDsCovarianceCalculator()

        C = \
            led_cov_calculator.compute_covariance(
                "shape", d_eye_param_d_data, d_eye_param_d_data_indices
            )

        self.assertTrue(sensitivity_analysis.is_valid_covariance(C))
