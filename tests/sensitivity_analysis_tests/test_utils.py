"""Common components to all sensitivity-analysis tests.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.scene as scene
import seet.sensitivity_analysis.derivative_calculators as derivative_calc
import unittest


class TestCommonUtils(unittest.TestCase):
    """Define common setup to sensitivity-analysis tests.
    """

    def setUp(self):
        super().setUp()

        # Create the default scene, computing gradients.
        self.et_scene = scene.SceneModel(requires_grad=True)
        self.derivative_data = derivative_calc.DataWrapper(self.et_scene)

    def extraSetup(self, cls, num_pupil_points=30, num_limbus_points=30):
        """Extra setup that depends on the specific object under testing.
        """

        self.derivative_calculator = cls(self.derivative_data)

    def _size_test(self, data_generator, derivative_method, M):
        """Tests for derivatives of glints using provided method.

        Args:
            data_generator (callable): method that generates the data for the
            computation of derivatives.

            derivative_method (callable): method in self.derivative_calculator
            object that computes derivative of the image feature with respect
            to relevant parameters.

            M (int): number of parameters with respect to which derivatives are
            computed.
        """

        data_generator()
        all_derivatives = derivative_method()
        for derivative in all_derivatives:
            # Image features are two dimensional.
            self.assertTrue(derivative.shape[0] == 2)

            # Check number of parameters with respect to which derivatives were
            # computed.
            self.assertTrue(derivative.shape[1] == M)


if __name__ == "__main__":
    unittest.main()
