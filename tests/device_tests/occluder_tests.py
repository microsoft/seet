"""occluder_tests.py

Unit tests for Occluder class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.device as device
import torch
import unittest


class TestOccluder(unittest.TestCase):
    """TestOccluder.

    Test Occluder class.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """

        super().setUp()

        self.fake_SubsystemModel = core.Node(name="fake ET subsystem")
        self.transform_soSubsystemModel_fromOccluder = \
            core.SE3.create_identity()

        self.occluder = \
            device.Occluder(
                self.fake_SubsystemModel,
                self.transform_soSubsystemModel_fromOccluder
            )

    def test_init(self):
        """test_init.

        Test initialization of occluder.
        """

        self.assertTrue(self.occluder.type == "window")

        self.assertTrue(self.occluder.coordinates.shape == torch.Size((3, 19)))

        self.assertTrue(
            torch.allclose(
                self.occluder.coordinates[:, 0],
                torch.tensor([15.6092, 8.93523, 11.9395])
            )
        )

    def test_mirror(self):
        """test_mirror.

        Test mirroring of occluder.
        """

        mirror_occluder = device.Occluder.mirror(self.occluder)

        self.assertTrue(
            torch.allclose(
                mirror_occluder.coordinates[:, 0],
                torch.tensor([-15.6092, 8.93523, 11.9395])
            )
        )

    def test_is_ray_occluded_inParent(self):
        """test_is_ray_occluded_inParent.

        Test occlusion test for ray in parent.
        """

        # Create a point in the center of the coordinates of the occluder, in
        # the coordinate system of the occluder parent.
        origin_inOccluder = torch.mean(self.occluder.coordinates, 1)
        origin_inParent = \
            self.occluder.transform_toParent_fromSelf.transform(
                origin_inOccluder
            )

        # Create a direction orthogonal to a plane near the occluders.
        u_inOccluder = self.occluder.coordinates[:, 0] - origin_inOccluder
        v_inOccluder = self.occluder.coordinates[:, 1] - origin_inOccluder
        normal_inOccluder = torch.cross(v_inOccluder, u_inOccluder)
        normal_inParent = \
            self.occluder.transform_toParent_fromSelf.rotation.transform(
                normal_inOccluder
            )

        self.assertFalse(
            self.occluder.is_ray_occluded_inParent(
                origin_inParent, normal_inParent
            )
        )

        # Same ray should be occluded in mirror occluder.
        mirror_occluder = device.Occluder.mirror(self.occluder)
        self.assertTrue(
            mirror_occluder.is_ray_occluded_inParent(
                origin_inParent, normal_inParent
            )
        )
