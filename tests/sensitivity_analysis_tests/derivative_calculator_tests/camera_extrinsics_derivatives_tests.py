"""Tests for sensitivity analysis of camera extrinsic parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from sensitivity_analysis import CameraExtrinsicsDerivatives
from tests.sensitivity_analysis_tests import test_utils
import unittest


class TestCameraExtrinsicsDerivatives(test_utils.TestCommonUtils):
    """Unit tests for sensitivity analysis of led locations.
    """

    def setUp(self):
        """Generate data for tests.
        """

        super().setUp()

        self.extraSetup(CameraExtrinsicsDerivatives)
        self.M = 6  # 3 for rotation, 3 for translation

    def test_size_compute_d_glints_d_parameters(self):
        """Test derivatives of glints with respect to camera extrinsics.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_glints_inPixels,
            self.derivative_calculator.compute_d_glints_d_parameters,
            self.M
        )

    def test_size_compute_d_pupil_d_parameters(self):
        """Test derivatives of pupil with respect to camera extrinsics.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.generate_pupil_inPixels,
            self.derivative_calculator.compute_d_pupil_d_parameters,
            self.M
        )

    def test_size_compute_d_limbus_d_parameters(self):
        """Test derivatives of limbus with respect to LED locations.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_limbus_inPixels,
            self.derivative_calculator.compute_d_limbus_d_parameters,
            self.M
        )


if __name__ == "__main__":
    unittest.main()
