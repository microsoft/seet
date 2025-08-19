"""groups.py.

Matrix groups.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.core import geometry
from seet.core import numeric
import torch


class PSL3():
    """PSL3.

    Class for creating and manipulating elements of the projective special
    linear group in 3 dimensions (PSL(3)). Derives from object so we can use
    Python 3 new-style classes.
    """

    @classmethod
    def is_valid_psl3(cls, matrix):
        """is_valid_psl3.

        Check whether the matrix is a valid representation of an element of
        PSL(3). Elements of PSL(3) can be represented by 4 x 4 real matrices
        with positive determinants.

        Args:
            matrix (torch.Tensor): object to be tested.
        """
        try:
            M, N = matrix.shape
        except ValueError("Input is not a 3 x 3 tensor."):  # type: ignore
            return False

        if M != 4 or N != 4:
            return False

        if torch.linalg.det(matrix) <= 0:
            return False

        return True

    @classmethod
    def compose_transforms(cls, transform_toA_fromB, transform_toB_fromC):
        """compose_transforms.

        Composition of elements of matrix groups.

        Parameters
        ----------
        transform_toA_fromB : PSL3
            Object representing transformation to coordinate system A from
            coordinate system B.

        transform_toB_fromC : PSL3
            Object representing transformation to coordinate system B from
            coordinate system C.

        Returns
        -------
        PSL3
            Object representing a transformation to coordinate system A from
            coordinate system C.
        """
        transform_matrix_toA_fromC = \
            transform_toA_fromB.transform_matrix @ \
            transform_toB_fromC.transform_matrix

        # Matrix multiplication is cheaper and more accurate than matrix
        # inversion.
        transform_matrix_toC_fromA = \
            transform_toB_fromC.inverse_transform_matrix @ \
            transform_toA_fromB.inverse_transform_matrix

        return cls(
            transform_matrix_toA_fromC,
            inverse_transform_matrix=transform_matrix_toC_fromA
        )

    @classmethod
    def create_identity(cls, requires_grad=False):
        Identity = torch.eye(4, requires_grad=requires_grad)
        if requires_grad:
            # Have to explicitly compute inverse so that derivatives are
            # correctly propagated.
            inv_Identity = torch.linalg.pinv(Identity)
        else:
            inv_Identity = torch.eye(4)

        return cls(Identity, inv_Identity)

    def __init__(
            self,
            transform_matrix,
            inverse_transform_matrix=None,
            requires_grad=False):
        """
        Create an element of PSL(3) from a 4 x 4 matrix.

        Args:
            transform_matrix (torch.Tensor): Tensor with shape 4 x 4
            representing an element of PSL(3).

            inverse_transform_matrix (torch.Tensor, optional): if provided,
            inverse of transform_matrix. If the inverse is known, providing it
            avoids an expensive matrix inversion.
        """

        # Verify whether the transform matrix is valid for PSL3.
        assert (PSL3.is_valid_psl3(transform_matrix)), \
            "Matrix is not a valid representation of an element of PSL(3)."

        self.transform_matrix = transform_matrix

        if inverse_transform_matrix is None:
            inverse_transform_matrix = torch.inverse(
                self.transform_matrix)
        else:
            assert (PSL3.is_valid_psl3(inverse_transform_matrix)), \
                "Inverse matrix is not a valid representation of an element" \
                + " of PLS(3)."

        self.inverse_transform_matrix = inverse_transform_matrix

    def create_inverse(self):
        """create_inverse.

        Create inverse element of group.

        Returns:
            PSL3: Inverse element, in the same subgroup as self.
        """
        # Using self.__class__ we ensure that the inverse is in the same group
        # as the calling object.
        return self.__class__(
            self.inverse_transform_matrix, self.transform_matrix
        )

    def homogeneous_transform(self, points_3D_h):
        """homogeneous_transform.

        Apply group action to points already in homogeneous coordinates.

        Args:
            points_3D_h (torch.Tensor): (4, ...)-shaped torch tensor.

        Returns:
            torch.Tensor: (4, ...)-shaped torch tensor
        """
        return self.transform_matrix @ points_3D_h

    def inverse_homogeneous_transform(self, points_3D_h):
        """inverse_homogeneous_transform.

        Apply inverse group action to points already in homogeneous
        coordinates.

        Args:
            points_3D_h (torch.Tensor): (4, ...)-shaped torch tensor.

        Returns:
            torch.Tensor: (4, ...)-shaped torch tensor
        """
        return self.inverse_transform_matrix @ points_3D_h

    def transform(self, points_3D):
        """transform.

        Transformation in cartesian coordinates"""
        result_h = self.homogeneous_transform(geometry.homogenize(points_3D))

        return geometry.dehomogenize(result_h)

    def inverse_transform(self, points_3D):
        result_h = self.inverse_homogeneous_transform(
            geometry.homogenize(points_3D))

        return geometry.dehomogenize(result_h)


class SO3(PSL3):
    @classmethod
    def is_valid_so3(cls, matrix):
        return geometry.is_valid_rotation(matrix, tol=numeric.EPS * 100)

    def __init__(self, transform_matrix, inverse_transform_matrix=None):
        """__init__.

        Create an element of SO(3)

        Args:
            transform_matrix (torch.Tensor): 3 x 3, 3 x 4, or 4 x 4 tensor with
            upper-right 3 x 3 block containing a rotation matrix. If 3 x 4,
            last column must be zero. If 4 x 4, top right 3 x 1 elements and
            bottom left 1 x 3 elements must be zero and bottom right 1 x 1
            element must be one. Alternatively, it may be a 3 or 3 x 1 vector,
            in which case it represents the rotation matrix in axis-angle form.

            inverse_transform_matrix (torch.Tensor, optional): If provided,
            this is the inverse of transform_matrix. If transform_matrix is 3 x
            4, this matrix is also 3 x 4 and its 3 x 3 left-side block is the
            inverse of the 3 x 3 left-side block of transform_matrix, with
            zeros in the last column.
        """

        if transform_matrix.ndim == 1 or transform_matrix.shape[1] == 1:
            R = geometry.rotation_matrix(transform_matrix)
        else:
            R = transform_matrix[:3, :3]
            assert (SO3.is_valid_so3(R)), "Not a valid rotation matrix."

        if inverse_transform_matrix is not None:
            R_inv = inverse_transform_matrix[:3, :3]
        else:
            R_inv = R.T

        actual = R_inv @ R
        expected = torch.eye(3)
        if not torch.allclose(
            actual, expected, rtol=numeric.EPS * 100, atol=numeric.EPS * 100
        ):
            print("Rotation matrix has low accuracy.")

        zero_vector = torch.zeros(3)
        one_vector = geometry.homogenize(zero_vector)

        three_by_four = torch.hstack((R, zero_vector.view(3, 1)))
        rotation_matrix = torch.vstack((three_by_four, one_vector.view(1, 4)))

        inverse_three_by_four = torch.hstack((R_inv, zero_vector.view(3, 1)))
        inverse_rotation_matrix = \
            torch.vstack((inverse_three_by_four, one_vector.view(1, 4)))

        super().__init__(rotation_matrix, inverse_rotation_matrix)

    def get_rotation_matrix(self):
        """get_rotation_matrix.

        Gets the 3 x 3 rotation matrix corresponding to the SO(3) element as a
        torch Tensor

        Returns:
            torch.Tensor: 3 x 3 torch tensor corresponding to the rotation
            matrix representing the SO(3) element.
        """
        return self.transform_matrix[:3, :3]


class SE3(PSL3):
    @classmethod
    def is_valid_se3(cls, matrix):
        try:
            M, N = matrix.shape
        except ValueError:
            print("Input is ill formed.")
            return False

        if M == 4:
            if N != 4:
                # Ill formed.
                return False

            if not torch.allclose(
                matrix[-1, :],
                geometry.homogenize(torch.zeros(3)),
                rtol=numeric.EPS,
                atol=numeric.EPS
            ):
                return False

        return SO3.is_valid_so3(matrix[:3, :3])

    def __init__(self, transform_matrix, inverse_transform_matrix=None):
        """__init__.

        Create element of SE(3)

        Parameters
        ----------
        transform_matrix : Tensor
            3, 3 x 1, 3 x 3, 3 x 4, or 4 x 4 tensor.

        inverse_transform_matrix : Tensor, optional
            If provided, should be the inverse of transform_matrix. If
            transform_matrix is 3 x 4, inverse_transform_matrix is the 3 x 4
            upper block of the inverse of the matrix obtained by vertically
            stacking transform_matrix to the 1 x 4 vector [0, 0, 0, 1].
        """
        if transform_matrix.ndim == 1 or transform_matrix.size()[1] == 1:
            # Input is translation.
            self.rotation = SO3(torch.eye(3))
            top = torch.hstack((torch.eye(3), transform_matrix.view(3, 1)))
            neg_top = \
                torch.hstack((torch.eye(3), -transform_matrix.view(3, 1)))
            bottom = torch.tensor((0.0, 0.0, 0.0, 1.0))
            transform_matrix_4x4 = torch.vstack((top, bottom.view(1, 4)))
            inverse_transform_matrix_4x4 = \
                torch.vstack((neg_top, bottom.view(1, 4)))
        else:
            M, N = transform_matrix.size()
            if N == 3:  # Input is 3 x 3
                assert (M == 3), \
                    "Transformation matrix must be 3 x 3, 3 x 4, or 4 x 4."

                # Input is a rotation matrix. We ignore
                # inverse_transform_matrix even if it is provided, since so3
                # efficiently computes the inverse via transposition.
                self.rotation = SO3(transform_matrix)
                transform_matrix_4x4 = self.rotation.transform_matrix
                inverse_transform_matrix_4x4 = \
                    self.rotation.inverse_transform_matrix
            else:
                if M == 3:  # Input is 3 x 4
                    assert (N == 4), \
                        "Transformation matrix must be 3 x 3, 3 x 4, or 4 x 4."

                    one_vector = geometry.homogenize(torch.zeros(3))
                    transform_matrix_4x4 = \
                        torch.vstack((transform_matrix, one_vector.view(1, 4)))
                    if inverse_transform_matrix is not None:
                        inverse_transform_matrix_4x4 = \
                            torch.vstack(
                                (
                                    inverse_transform_matrix,
                                    one_vector.view(1, 4)
                                )
                            )
                    else:
                        inverse_transform_matrix_4x4 = None
                else:  # Input is 4 x 4
                    assert (M == 4 and N == 4), \
                        "Transformation matrix must be 3 x 3, 3 x 4, or 4 x 4."
                    transform_matrix_4x4 = transform_matrix
                    inverse_transform_matrix_4x4 = inverse_transform_matrix

                self.rotation = SO3(transform_matrix_4x4[:3, :3])

        super().__init__(transform_matrix_4x4, inverse_transform_matrix_4x4)
