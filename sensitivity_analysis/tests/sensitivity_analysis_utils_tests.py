"""Test utilities for sensitivity analysis.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import sensitivity_analysis as sensitivity_analysis
from parameterized import parameterized
import torch
import unittest


class TestSensitivityAnalysisUtils(unittest.TestCase):
    """Unit tests for sensitivity-analysis utilities.
    """

    def setUp(self):
        """Initialize data for tests.
        """

        super().setUp()

        self.std = torch.tensor([2.0, 3.0, 1.0])

    @parameterized.expand(
        [
            (None, ),
            (torch.tensor([0.1, -0.2, 0.1]), )
        ]
    )
    def test_build_single_cov_matrix(self, intra_cor):
        """Test building of covariance matrix.

        Args:
            intra_cor (None or torch.Tensor) inner cross-covariance terms. If
            None, the inner cross-covariance is assumed to be zero.
        """

        C = \
            sensitivity_analysis.build_single_cov_matrix(
                self.std,
                intra_cor=intra_cor
            )

        self.assertTrue(sensitivity_analysis.is_valid_covariance(C))

        # If intra_cor is None, is diagonal.
        if intra_cor is None:
            self.assertTrue(torch.allclose(C, torch.diag(torch.diag(C))))

    @parameterized.expand(
        [
            (None, ),
            (0.0, ),
            (0.5, )
        ]
    )
    def test_build_cross_cov_matrix(self, inter_cor):
        """Test building of cross-covariance matrix.

        Args:
            inter_cor (float or None): cross-correlation value.
        """

        S = sensitivity_analysis.build_single_cov_matrix(self.std)
        C = sensitivity_analysis.build_cross_cov_matrix(S, inter_cor=inter_cor)

        # Is square
        self.assertTrue(C.shape[0] == C.shape[1])

        # Has the right size.
        self.assertTrue(C.shape[0] == len(S))
