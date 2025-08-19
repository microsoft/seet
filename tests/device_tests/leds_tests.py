"""leds_tests.py

Unit tests for LEDs class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.device as device
import torch
import unittest


class TestLEDs(unittest.TestCase):
    """TestLEDs.

    Test class LEDs.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """
        super().setUp()

        self.fake_SubsystemModel = core.Node(name="fake ET subsystem")
        self.transform_toSubsystemModel_fromLEDs = core.SE3.create_identity()

        # Let's create the habit of calling instances of LEDs "led_set", to
        # avoid shadowing leds or LEDs.
        self.led_set = \
            device.LEDs(
                self.fake_SubsystemModel,
                self.transform_toSubsystemModel_fromLEDs
            )

    def test_init(self):
        """test_init.

        Test initialization of LEDs.
        """

        self.assertTrue(
            self.led_set.coordinates.shape == torch.Size((3, 12))
        )

        self.assertTrue(
            torch.allclose(
                self.led_set.coordinates[:, 0],
                torch.tensor([54.2615, 0.845241, 23.9997])
            )
        )

    def test_mirror(self):
        """test_mirror.

        Test mirroring of LEDs.
        """

        mirror_set = device.LEDs.mirror(self.led_set)

        self.assertTrue(
            torch.allclose(
                mirror_set.coordinates[:, 0],
                torch.tensor([-54.2615, 0.845241, 23.9997])
            )
        )


if __name__ == "__main__":
    unittest.main()
