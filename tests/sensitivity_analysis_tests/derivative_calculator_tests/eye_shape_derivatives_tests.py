"""eye_shape_sensitivity_test.py.

Tests for sensitivity analysis of eye-shape parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from sensitivity_analysis import EyeShapeDerivatives
from tests.sensitivity_analysis_tests import test_utils
import unittest


class TestEyeShapeDerivatives(test_utils.TestCommonUtils):
    """Unit tests for sensitivity analysis of eye-shape parameters.
    """

    def setUp(self):
        """setUp.

        Generate data for tests.
        """

        super().setUp()

        self.extraSetup(EyeShapeDerivatives)
        self.M = 9  # There are 9 eye-shape parameters.

    def test_size_compute_d_glints_d_parameters(self):
        """Test derivatives of measurements with respect to eye-shape
        parameters.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_glints_inPixels,
            self.derivative_calculator.compute_d_glints_d_parameters,
            self.M
        )

    def test_size_compute_d_pupil_d_parameters(self):
        """Test derivatives of pupil with respect to eye-shape parameters.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.generate_pupil_inPixels,
            self.derivative_calculator.compute_d_pupil_d_parameters,
            self.M
        )

    def test_size_compute_d_limbus_d_parameters(self):
        """Test derivatives of limbus with respect to eye-shape parameters.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_limbus_inPixels,
            self.derivative_calculator.compute_d_limbus_d_parameters,
            self.M
        )


if __name__ == "__main__":
    unittest.main()
