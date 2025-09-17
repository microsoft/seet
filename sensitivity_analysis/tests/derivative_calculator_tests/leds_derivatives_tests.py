"""Tests for sensitivity analysis of LED locations.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from sensitivity_analysis import LEDLocationsDerivatives
from sensitivity_analysis.tests import test_utils
import unittest


class TestLEDLocationsDerivatives(test_utils.TestCommonUtils):
    """Unit tests for sensitivity analysis of led locations.
    """

    def setUp(self):
        """Generate data for tests.
        """

        super().setUp()

        self.extraSetup(LEDLocationsDerivatives)
        # Each LED has 3 coordinates.
        self.M_leds = self.derivative_calculator.get_num_parameters()
        self.M_transform = 6

    def test_size_compute_d_glints_d_parameters(self):
        """Test derivatives of glints with respect to LED locations.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_glints_inPixels,
            self.derivative_calculator.compute_d_glints_d_parameters,
            self.M_leds
        )

    def test_size_compute_d_pupil_d_parameters(self):
        """Test derivatives of pupil with respect to LED locations.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.generate_pupil_inPixels,
            self.derivative_calculator.compute_d_pupil_d_parameters,
            self.M_leds
        )

    def test_size_compute_d_limbus_d_parameters(self):
        """Test derivatives of limbus with respect to LED locations.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_limbus_inPixels,
            self.derivative_calculator.compute_d_limbus_d_parameters,
            self.M_leds
        )

    def test_size_compute_d_glints_d_leds_transformation(self):
        """Test derivatives of glints with respect to global LED locations.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_glints_inPixels,
            self.derivative_calculator.compute_d_glints_d_leds_transformation,
            self.M_transform
        )

    def test_size_compute_d_pupil_d_leds_transformation(self):
        """Test derivatives of pupil with respect to global LED locations.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.generate_pupil_inPixels,
            self.derivative_calculator.compute_d_pupil_d_leds_transformation,
            self.M_transform
        )

    def test_size_compute_d_limbus_d_leds_transformation(self):
        """Test derivatives of limbus with respect to global LED locations.
        """

        self._size_test(
            self.derivative_calculator.derivative_data.
            generate_limbus_inPixels,
            self.derivative_calculator.compute_d_limbus_d_leds_transformation,
            self.M_transform
        )


if __name__ == "__main__":
    unittest.main()
