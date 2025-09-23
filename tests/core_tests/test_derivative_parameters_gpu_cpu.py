
import torch
import unittest
from sensitivity_analysis.derivative_calculators.derivative_parameters import LEDsParameters

class DummyParent:
    def get_transform_toOther_fromSelf(self, camera):
        class DummyTransform:
            def __init__(self):
                self.transform_matrix = torch.eye(4)  # or whatever shape is e
                self.inverse_transform_matrix = torch.eye(4)
            def create_inverse(self):
                return self
            def transform(self, led):
                return led
        return DummyTransform()

class DummyLEDSet:
    def __init__(self, num_leds=2):
        self.coordinates = torch.randn(3, num_leds)
        self.parent = DummyParent()
    def update_transform_toParent_fromSelf(self, T):
        pass

class DummySubsystem:
    def __init__(self, num_leds=2):
        self.cameras = [object()]
        self.led_set = DummyLEDSet(num_leds)

class TestLEDsParameters(unittest.TestCase):
    def setUp(self):
        self.num_leds = 3
        self.subsystem = DummySubsystem(num_leds=self.num_leds)

    def test_leds_parameters_cpu(self):
        #leds_params = LEDsParameters(self.subsystem)
        leds_params = LEDsParameters(self.subsystem, device=torch.device('cpu'))  # for CPU
        self.assertIsInstance(leds_params.leds.coordinates, torch.Tensor)
        self.assertEqual(leds_params.leds.coordinates.device.type, 'cpu')
        self.assertEqual(leds_params.leds.coordinates.shape[0], 3)
        self.assertEqual(leds_params.leds.coordinates.shape[1], self.num_leds)

    def test_leds_parameters_gpu(self):
        if torch.cuda.is_available():
            #leds_params = LEDsParameters(self.subsystem)
            leds_params = LEDsParameters(self.subsystem, device=torch.device('cuda')) # for GPU
            device = torch.device('cuda')
            leds_params.leds.coordinates = leds_params.leds.coordinates.to(device)
            self.assertEqual(leds_params.leds.coordinates.device.type, 'cuda')
            self.assertEqual(leds_params.leds.coordinates.shape[0], 3)
            self.assertEqual(leds_params.leds.coordinates.shape[1], self.num_leds)
        else:
            self.skipTest('CUDA not available')

if __name__ == '__main__':
    unittest.main()
