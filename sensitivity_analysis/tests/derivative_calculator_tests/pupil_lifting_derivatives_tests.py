"""Test for sensitivity analysis of LED locations.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from sensitivity_analysis import PupilLiftingDerivatives
from sensitivity_analysis.tests import test_utils
import torch
import unittest


class TestPupilLiftingDerivatives(test_utils.TestCommonUtils):
    """Unit tests for sensitivity analysis of lifting parameters.
    """

    def setUp(self):
        """Generate data for tests.
        """

        super().setUp()

        self.extraSetup(PupilLiftingDerivatives)

    def test_size_compute_d_glints_d_lifting_parameters(self):
        """Test size of derivatives of glints w.r.t. lifting parameters.
        """

        # Test of derivatives of glints with respect to pupil lifting
        # parameters requires computation of both glints and pupil.

        # We need to generate the data first.
        self.derivative_calculator.derivative_data.generate_glints_inPixels()
        self.derivative_calculator.derivative_data.generate_pupil_inPixels()

        # The function signature requires a callable to generate the data, but
        # we have generated the data already. Pass a fake callable instead.
        def fake_data_generator():
            pass

        self._size_test(
            fake_data_generator,
            self.derivative_calculator.compute_d_glints_d_parameters,
            self.derivative_calculator.get_num_parameters()
        )

    def test_compute_d_glints_d_lifting_parameters(self):
        """Test derivatives of glints with respect to lifting parameters.
        """

        self.derivative_calculator.derivative_data.generate_glints_inPixels()
        self.derivative_calculator.derivative_data.generate_pupil_inPixels()

        d_glints_d_parameters = \
            self.derivative_calculator.compute_d_glints_d_parameters()

        for d_glint_i_d_lifting_parameters in d_glints_d_parameters:
            if d_glint_i_d_lifting_parameters is None:
                continue

            self.assertTrue(
                torch.allclose(
                    d_glint_i_d_lifting_parameters,
                    torch.zeros_like(d_glint_i_d_lifting_parameters)
                )
            )

    def test_compute_d_pupil_d_parameters(self):
        """Test derivatives of pupil points with respect to lifting parameters.
        """

        self.derivative_calculator.derivative_data.generate_pupil_inPixels()

        derivative_calculator = self.derivative_calculator
        d_pupil_d_parameters = \
            derivative_calculator.compute_d_pupil_d_parameters()

        # Derivative is zero everywhere except for the exact pair pupil-point
        # and lifting parameter/pupil-point angle.
        num_cols_minus_one = derivative_calculator.get_num_parameters() - 1
        for i, d_pupil_point_d_parameters in enumerate(d_pupil_d_parameters):
            if d_pupil_point_d_parameters is None:
                continue

            cols = list(torch.split(d_pupil_point_d_parameters, 1, dim=1))

            non_zero_part = cols.pop(i)
            self.assertFalse(
                torch.allclose(non_zero_part, torch.zeros((2, 1)))
            )

            zero_part = torch.hstack(cols)
            self.assertTrue(
                torch.allclose(zero_part, torch.zeros((2, num_cols_minus_one)))
            )

    def test_compute_d_limbus_d_lifting_parameters(self):
        """Test derivatives of limbus points w.r.t. lifting parameters.
        """

        self.derivative_calculator.derivative_data.generate_pupil_inPixels()
        self.derivative_calculator.derivative_data.generate_limbus_inPixels()

        derivative_calculator = self.derivative_calculator
        d_limbus_d_parameters = \
            derivative_calculator.compute_d_limbus_d_parameters()

        # All derivatives are zero.
        num_cols = derivative_calculator.get_num_parameters()
        for d_limbus_point_d_parameters in d_limbus_d_parameters:
            if d_limbus_point_d_parameters is None:
                continue

            self.assertTrue(
                torch.allclose(
                    d_limbus_point_d_parameters, torch.zeros((2, num_cols))
                )
            )


if __name__ == "__main__":
    unittest.main()
