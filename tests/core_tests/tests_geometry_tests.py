"""geometry_tests.py.

Unit tests for methods and classes in core.py
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"

import seet.core as core
import torch
import unittest


class TestGeometry(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.tensor([1.0, 0, 0])
        self.y = torch.tensor([0.0, 1, 0])
        self.z = torch.tensor([0.0, 0, 1])
        self.basis = (self.x, self.y, self.z)
        self.theta_deg = 90.0

        return super().setUp()

    def test_rotation_around_axes(self):
        axes = {
            "x": (
                core.rotation_around_x,
                (self.x, self.z, -self.y)  # x to x, y to z, z to -y
            ),
            "y": (
                core.rotation_around_y,
                (-self.z, self.y, self.x)  # x to -z, y to y, z to x
            ),
            "z": (
                core.rotation_around_z,
                (self.y, -self.x, self.z)  # x to y, y to -x, z to z
            )
        }
        for val in axes.values():
            R = val[0](self.theta_deg)
            for i in range(3):
                to = R @ self.basis[i]
                to_ = val[1][i]
                self.assertTrue(
                    torch.allclose(
                        to, to_, rtol=core.EPS, atol=core.EPS
                    )
                )

    def test_get_yaw_pitch_angles_deg(self):
        angle_deg = torch.tensor(45.0)

        # Create unit vector with known yaw.
        v = core.rotation_around_y(angle_deg) @ self.z
        yaw_pitch_deg = core.get_yaw_pitch_angles_deg(v)

        # A positive rotation around y has negative yaw and no pitch.
        self.assertTrue(
            torch.allclose(
                -yaw_pitch_deg[0],
                angle_deg,
                rtol=core.EPS,
                atol=core.EPS
            )
        )
        self.assertTrue(
            torch.allclose(
                yaw_pitch_deg[1],
                core.T0,
                rtol=core.EPS,
                atol=core.EPS
            )
        )

        # Create a unit vector with know pitch.
        v = core.rotation_around_x(angle_deg) @ self.z
        yaw_pitch_deg = core.get_yaw_pitch_angles_deg(v)

        # A positive rotation around x has no yaw and negative pitch.
        self.assertTrue(
            torch.allclose(
                yaw_pitch_deg[0],
                core.T0,
                rtol=core.EPS,
                atol=core.EPS
            )
        )
        self.assertTrue(
            torch.allclose(
                -yaw_pitch_deg[1],
                angle_deg,
                rtol=core.EPS,
                atol=core.EPS
            )
        )

    def test_homogenize_dehomogenize_normalize(self):
        num_points = 10
        for dim in range(2, 5):
            cartesian = torch.tensor(
                range(dim * num_points),
                dtype=core.DTYPE
            ).view(dim, num_points)
            homogeneous = core.homogenize(cartesian)
            new_cartesian = core.dehomogenize(homogeneous)
            new_homogeneous = core.homogenize(new_cartesian)
            self.assertTrue(
                torch.allclose(
                    homogeneous,
                    new_homogeneous,
                    rtol=core.EPS,
                    atol=core.EPS
                )
            )

            normalized = core.normalize(new_cartesian)
            self.assertTrue(
                torch.allclose(
                    torch.linalg.norm(normalized, dim=0),
                    torch.ones(num_points, dtype=core.DTYPE),
                    rtol=core.EPS,
                    atol=core.EPS
                )
            )

    def test_expand_to_rotation_from_x_y_axes(self):
        # Rotations that exchange directions of axes.
        # 1. Rotation around x sends x to x, y to z, z to -y
        R = core.rotation_around_x(self.theta_deg)
        x_axis = self.x
        y_axis = self.z
        R_ = core.expand_to_rotation_from_x_y_axes(
            x_axis=x_axis,
            y_axis=y_axis
        )
        self.assertTrue(
            torch.allclose(R, R_, rtol=core.EPS, atol=core.EPS)
        )

        # 2. Rotation around y sends x to -z, y to y, z to x.
        R = core.rotation_around_y(self.theta_deg)
        x_axis = -self.z
        z_axis = self.x
        R_ = core.expand_to_rotation_from_x_y_axes(
            x_axis=x_axis,
            z_axis=z_axis
        )
        self.assertTrue(
            torch.allclose(R, R_, rtol=core.EPS, atol=core.EPS)
        )

        # 2. Rotation around z sends # x to y, y to -x, z to z.
        R = core.rotation_around_z(self.theta_deg)
        y_axis = -self.x
        z_axis = self.z
        R_ = core.expand_to_rotation_from_x_y_axes(
            y_axis=y_axis,
            z_axis=z_axis
        )
        self.assertTrue(
            torch.allclose(R, R_, rtol=core.EPS, atol=core.EPS)
        )

    def test_is_valid_rotation(self):
        R = core.rotation_around_x(self.theta_deg)
        self.assertTrue(core.is_valid_rotation(R))

        axis = torch.tensor(range(3), dtype=core.DTYPE)
        R = core.rotation_matrix(axis)
        self.assertTrue(core.is_valid_rotation(R))

    def test_enforce_rotation(self):
        # Create a matrix that is not a rotation.
        not_R = torch.diag(torch.tensor([1.0, 2, -1]))
        yes_R = core.enforce_rotation(not_R)
        self.assertTrue(
            torch.allclose(
                yes_R,
                torch.eye(3),
                rtol=core.EPS,
                atol=core.EPS)
        )

    def test_hat_vee_so3(self):
        x = torch.tensor(range(3), dtype=core.DTYPE)
        y = torch.tensor([2.0, -3, 4])

        hat_x = core.hat_so3(x)
        x_ = core.vee_so3(hat_x)
        cross = torch.cross(x, y)

        for cross_ in (hat_x @ y, torch.cross(x_, y)):
            self.assertTrue(
                torch.allclose(
                    cross,
                    cross_,
                    rtol=core.EPS,
                    atol=core.EPS
                )
            )

    def test_rotation_matrix(self):
        # 90 deg rotation around x
        axis = self.x * core.deg_to_rad(self.theta_deg)
        R = core.rotation_matrix(axis)
        R_ = core.rotation_around_x(self.theta_deg)

        self.assertTrue(
            torch.allclose(R, R_, rtol=core.EPS, atol=core.EPS)
        )

    def test_rotation_axis(self):
        # 90 deg rotation around y
        axis = self.y * core.deg_to_rad(self.theta_deg)
        R = core.rotation_around_y(self.theta_deg)
        axis_ = core.rotation_axis(R)

        self.assertTrue(
            torch.allclose(axis, axis_, rtol=core.EPS, atol=core.EPS)
        )

    def test_rotation_matrix_from_u_to_v(self):
        # 90 deg rotation around y sends z to x.
        R = core.rotation_matrix_from_u_to_v(self.z, self.x)
        R_ = core.rotation_around_y(self.theta_deg)

        self.assertTrue(
            torch.allclose(R, R_, rtol=core.EPS, atol=core.EPS)
        )


if __name__ == "__main__":
    unittest.main()
