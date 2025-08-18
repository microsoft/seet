"""device_model_tests.py

Unit tests for ET device. An ET device consists of one or more ET
subsystems, each consisting of LEDs and one or more cameras.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.device as device
from tests.device_tests import device_tests_configs
import os
import torch
import unittest


class TestDeviceModel(unittest.TestCase):
    """TestDeviceModel.

    Unit tests for ET device.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """

        super().setUp()

        self.device = \
            device.DeviceModel(
                core.Node(),
                core.SE3.create_identity(),
                requires_grad=True
            )

        self.another_device = \
            device.DeviceModel(
                core.Node(),
                core.SE3.create_identity(),
                parameter_file_name=os.path.join(
                    device_tests_configs.DEVICE_MODELS_TEST_DIR,
                    r"device_tests_data/test_device.json"
                ),
                requires_grad=False
            )

    def test_init(self):
        """test_init.

        Test initialization of ET device.
        """

        num_subsystems = 2
        num_led_coordinates = 3
        num_leds = 12
        num_cameras = 1
        for a_device in (self.device, self.another_device):
            # Devices have two ET subsystems.
            self.assertEqual(num_subsystems, len(a_device.subsystems))

            # LEDs and cameras on devices have the same parameters.
            for i in range(num_subsystems):
                # The subsystems have the same number of LEDs
                led_set = a_device.subsystems[i].led_set
                self.assertEqual(num_led_coordinates, len(led_set.coordinates))
                self.assertEqual(num_leds, len(led_set.coordinates[0]))

                # The subsystem have the same number of cameras.
                cameras = a_device.subsystems[i].cameras
                self.assertEqual(num_cameras, len(cameras))

    def test_mirror(self):
        """test_mirror.

        Test mirroring of ET subsystem.
        """
        # ET subsystems mirror each other.
        subsystem = self.device.subsystems[0]
        mirror_subsystem = self.device.subsystems[1]

        # Test mirroring of LEDs. Mirroring is with respect to yz plane of
        # device.
        led_coordinates_inOther = \
            subsystem.led_set.get_coordinates_inOther(self.device)
        mirror_led_coordinates_inOther = \
            mirror_subsystem.led_set.get_coordinates_inOther(self.device)

        self.assertTrue(
            torch.allclose(
                led_coordinates_inOther,
                torch.diag(
                    torch.tensor([-1.0, 1.0, 1.0])
                ) @ mirror_led_coordinates_inOther
            )
        )

        # Test mirroring of cameras. Mirroring is with respect to yz plane of
        # device.
        point_in3D = torch.tensor([1.0, 2.0, 5.0])
        camera = subsystem.camera
        mirror_camera = mirror_subsystem.camera

        point_inPixels = \
            camera.project_toPixels_fromOther(point_in3D, self.device)
        mirror_point_in3D = torch.tensor([-1.0, 1.0, 1.0]) * point_in3D
        mirror_point_inPixels = \
            mirror_camera.project_toPixels_fromOther(
                mirror_point_in3D, self.device
            )
        mirror_point_inImage = \
            mirror_camera.get_point_inImagePlane(mirror_point_inPixels)
        point_inImage_ = torch.tensor([-1.0, 1.0]) * mirror_point_inImage
        point_inPixels_ = \
            mirror_camera.get_point_inPixels(point_inImage_)

        self.assertTrue(torch.allclose(point_inPixels, point_inPixels_))


if __name__ == "__main__":
    unittest.main()
