import sys
sys.path.append('../src/')
import unittest
import torch
from models import PanUNet, resnet18

expected_size = (2, 3, 512, 512)

class TestModel(unittest.TestCase):
    """Test of model
    """
    def test_unet(self):
        net = PanUNet(3, 3)
        inputs1 = torch.rand(2, 3, 256, 256)
        inputs2 = torch.rand(2, 3, 512, 512)
        outputs = net(inputs1, inputs2)
        self.assertEqual(outputs.size(), expected_size)

    def test_resnet(self):
        inputs1 = torch.rand(2, 3, 128, 128)
        inputs2 = torch.rand(2, 1, 128, 128)
        net = resnet18(4, 3).eval()
        outputs = net(inputs1, inputs2)
        self.assertEqual(inputs1.size(), outputs.size())
        net = resnet18(4, 3, batch_norm=False).eval()
        outputs = net(inputs1, inputs2)
        self.assertEqual(inputs1.size(), outputs.size())

if __name__ == '__main__':
    unittest.main()
