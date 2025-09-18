"""subsystem_model_tests.py

Unit tests for ET subsystem class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.device as device
import torch
import unittest
import os


class TestSubsystemModel(unittest.TestCase):
    """TestSubsystemModel.

    Unit tests for ET subsystem.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """

        super().setUp()

        self.subsystem = device.SubsystemModel(
            core.Node(),
            core.SE3.create_identity(),
            requires_grad=True
        )

    def test_init(self):
        """test_init.

        Test initialization of ET subsystem.
        """

        # There's a single camera:
        self.assertTrue(
            self.subsystem.cameras[0] == self.subsystem.cameras[-1]
        )
        self.assertTrue(self.subsystem.camera == self.subsystem.cameras[-1])

        # We got the camera matrix right.
        camera = self.subsystem.camera
        self.assertTrue(
            torch.allclose(
                camera.transform_toParent_fromSelf.inverse_transform_matrix,
                torch.tensor(
                    [
                        [1.0000e+00, 7.5557e-08, 4.3975e-08, -3.0000e+01],
                        [8.7423e-08, -8.6427e-01, -5.0302e-01, 4.8619e-01],
                        [0.0000e+00, 5.0302e-01, -8.6427e-01, 4.0475e+01],
                        [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]
                    ]
                ),
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

        self.assertTrue(
            self.subsystem.led_set.coordinates.shape ==
            torch.Size((3, 12))
        )

        self.assertTrue(
            torch.allclose(
                self.subsystem.led_set.coordinates[:, 0],
                torch.tensor([54.2615, 0.845241, 23.9997])
            )
        )

    def test_mirror(self):
        """test_mirror.

        Test mirroring of ET subsystem.
        """

        mirror_subsystem = \
            device.SubsystemModel.mirror(self.subsystem)

        mirror_led_set = mirror_subsystem.led_set

        self.assertTrue(
            torch.allclose(
                mirror_led_set.coordinates[:, 0],
                torch.tensor([-54.2615, 0.845241, 23.9997])
            )
        )

        # Mirroring happens in the normalized image plane. Its effect on the
        # pixel values is different.
        point_in3D = torch.tensor([1.0, 2.0, 5.0])
        camera = self.subsystem.camera
        point_inPixels = camera.project_toPixels_fromCamera(point_in3D)

        mirror_camera = camera.mirror(camera)

        mirror_point_in3D = torch.tensor([-1.0, 1.0, 1.0]) * point_in3D
        mirror_point_inPixels = \
            mirror_camera.project_toPixels_fromCamera(mirror_point_in3D)
        mirror_point_inImage = \
            mirror_camera.get_point_inImagePlane(mirror_point_inPixels)
        point_inImage_ = torch.tensor([-1.0, 1.0]) * mirror_point_inImage
        point_inPixels_ = \
            mirror_camera.get_point_inPixels(point_inImage_)

        self.assertTrue(torch.allclose(point_inPixels, point_inPixels_))

    def test_apply_occluder_inSubsystem(self):
        """test_apply_occluder_inSubsystem.

        Test whether occluder really occludes points.
        """

        # Create a subsystem that uses the device with occluder
        subsystem_with_occluder = device.SubsystemModel(
            core.Node(),
            core.SE3.create_identity(),
            parameter_file_name=os.path.join(
                device.DEVICE_DIR,
                "default_device/default_with_occluder_left_subsystem_model.json"
            ),
            requires_grad=True
        )

        # Pylance complains otherwise...
        assert (subsystem_with_occluder.occluder is not None)

        # Create a ray that is not occluded.
        occluder_coordinates_inSubsystem = \
            subsystem_with_occluder.occluder.get_coordinates_inParent()

        reference_inSubsystem = occluder_coordinates_inSubsystem.mean(axis=1)
        points_inSubsystem = \
            occluder_coordinates_inSubsystem + \
            torch.tensor([[0.0], [0.0], [10.0]])
        points_inSubsystem = list(points_inSubsystem.T)
        visible_inSubsystem = \
            subsystem_with_occluder.apply_occluder_inSubsystem(
                points_inSubsystem, reference_inSubsystem
            )

        for pt, visible_pt in zip(points_inSubsystem, visible_inSubsystem):
            self.assertTrue(torch.allclose(pt, visible_pt))

        # Create rays that are occluded.
        reference_inSubsystem = \
            reference_inSubsystem + torch.tensor([30.0, 0.0, 0.0])
        invisible_inSubsystem = \
            subsystem_with_occluder.apply_occluder_inSubsystem(
                points_inSubsystem, reference_inSubsystem
            )

        for invisible_pt in invisible_inSubsystem:
            self.assertTrue(invisible_pt is None)


if __name__ == "__main__":
    unittest.main()
