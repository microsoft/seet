"""Unit tests for methods and classes in core.py.

Unit tests for methods and classes in core.py
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import torch
import unittest


class TestGroups(unittest.TestCase):
    """TestGroups.

    Unit tests for Lie core.

    """

    def setUp(self) -> None:
        self.bad_shape = torch.tensor(
            [[1.0, 2, 3],
             [-1, 0, 3]]
        )

        self.bad_determinant = torch.diag(
            torch.tensor([1.0, -1.0, 1.0, 1.0])
        )

        axis = torch.tensor([1.0, -2, 3])
        self.rotation_matrix = core.rotation_matrix(axis)

        self.positive_determinant_4x4 = torch.tensor(
            [[4.00, 0.30, -0.10, 1.20],
             [-0.20, 5.00, 0.20, -3.50],
             [0.10, -0.30, 2.00, 4.30],
             [0.03, 0.20, -0.50, 1.00]]
        )
        assert (torch.linalg.det(self.positive_determinant_4x4) > 0), \
            "4 x 4 matrix in test does not have positive determinant."

        self.element = core.PSL3(self.positive_determinant_4x4)
        self.inverse_element = core.PSL3(
            torch.linalg.pinv(self.positive_determinant_4x4)
        )

        return super().setUp()

    def test_is_valid_psl3(self):
        # Bad shape.
        self.assertFalse(core.PSL3.is_valid_psl3(self.bad_shape))

        # Good shape, bad determinant.
        self.assertFalse(core.PSL3.is_valid_psl3(self.bad_determinant))

        # Good shape, good determinant.
        self.assertTrue(core.PSL3.is_valid_psl3(
            self.positive_determinant_4x4))

    def test_compose_transform_create_identity(self):
        id = core.PSL3.compose_transforms(self.element, self.inverse_element)
        id_ = core.PSL3.create_identity()

        device = id.transform_matrix.device
        self.assertTrue(
            torch.allclose(
                id.transform_matrix,
                id_.transform_matrix.to(device),
                rtol=core.EPS * 100,  # We need to be forgiving
                atol=core.EPS * 100  # because errors accumulate.
            )
        )

    def test_create_inverse(self):
        inverse_element_ = self.element.create_inverse()

        self.assertTrue(
            torch.allclose(
                self.inverse_element.transform_matrix,
                inverse_element_.transform_matrix,
                rtol=core.EPS * 100,  # We need to be forgiving
                atol=core.EPS * 100  # because errors accumulate.
            )
        )

    def test_transform_homogeneous_transform(self):
        cartesian = torch.tensor([1.0, 2.0, 3.0])
        # Commutative diagram
        #
        #          T
        #   x------------>  T_x
        #   |                |
        # h |                | h
        #   |                |
        #   V                V
        #  x_h -----------> T_x_h
        #         T_h

        # Counterclockwise using inverse element.
        homogeneous = core.homogenize(cartesian)
        T_homogeneous = self.element.homogeneous_transform(homogeneous)
        T_cartesian = core.dehomogenize(T_homogeneous)
        # Use inverse element.
        cartesian_ = self.inverse_element.transform(T_cartesian)

        self.assertTrue(torch.allclose(cartesian, cartesian_))

        # Clockwise using inverse element.
        T_cartesian = self.element.transform(cartesian)
        T_homogeneous = core.homogenize(T_cartesian)
        # Use inverse element.
        homogeneous = self.inverse_element.homogeneous_transform(T_homogeneous)
        cartesian_ = core.dehomogenize(homogeneous)

        self.assertTrue(torch.allclose(cartesian, cartesian_))

        # Counterclockwise using inverse transform.
        homogeneous = core.homogenize(cartesian)
        T_homogeneous = self.element.homogeneous_transform(homogeneous)
        T_cartesian = core.dehomogenize(T_homogeneous)
        # Use inverse transform
        cartesian_ = self.element.inverse_transform(T_cartesian)

        self.assertTrue(torch.allclose(cartesian, cartesian_))

        # Clockwise using inverse transform.
        T_cartesian = self.element.transform(cartesian)
        T_homogeneous = core.homogenize(T_cartesian)
        # Use inverse trasform.
        homogeneous = self.element.inverse_homogeneous_transform(T_homogeneous)
        cartesian_ = core.dehomogenize(homogeneous)

        self.assertTrue(torch.allclose(cartesian, cartesian_))

    def test_is_valid_so3(self):
        # Bad shape.
        self.assertFalse(core.SO3.is_valid_psl3(self.bad_shape))

        # Good shape, bad determinant.
        self.assertFalse(core.SO3.is_valid_psl3(self.bad_determinant))

        # Good shape, good determinant, R.T @ R == I
        self.assertTrue(core.SO3.is_valid_so3(self.rotation_matrix))

    def test_get_rotation_matrix(self):
        element = core.SO3(self.rotation_matrix)
        rotation_matrix = element.get_rotation_matrix()

        self.assertTrue(
            torch.allclose(
                self.rotation_matrix,
                rotation_matrix,
                rtol=core.EPS,
                atol=core.EPS
            )
        )

        self.assertTrue(core.is_valid_rotation(rotation_matrix))

    def test_is_valid_se3(self):
        matrix = torch.tensor(
            [[4.00, 0.30, -0.10, 1.20],
             [-0.20, 5.00, 0.20, -3.50],
             [0.10, -0.30, 2.00, 4.30],
             [0.03, 0.20, -0.50, 1.00]]
        )
        self.assertFalse(core.SE3.is_valid_se3(matrix))

        not_R = matrix[:3, :3]
        yes_R = core.enforce_rotation(not_R)
        zeros_3x1 = torch.zeros(3, 1)
        matrix = torch.hstack((yes_R, zeros_3x1))
        matrix = torch.vstack((matrix, core.homogenize(zeros_3x1).T))

        self.assertTrue(core.SE3.is_valid_se3(matrix))

        translation_vector = torch.tensor([1.0, 2.0, 3.0])
        translation_transform = core.SE3(translation_vector)
        self.assertTrue(
            core.SE3.is_valid_se3(translation_transform.transform_matrix)
        )


if __name__ == "__main__":
    unittest.main()
