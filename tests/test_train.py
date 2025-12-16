"""
Unit tests for the Heart Segmentation training pipeline.

These tests verify that the model can:
1. Initialize correctly
2. Perform a forward pass with the expected output shape
3. Compute loss without errors
"""
import unittest
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose


class TestModel(unittest.TestCase):
    """Test that the UNet model works correctly."""

    @classmethod
    def setUpClass(cls):
        """Set up model and loss for all tests."""
        cls.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        cls.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(cls.device)
        cls.loss_function = DiceLoss(
            to_onehot_y=True, sigmoid=True, squared_pred=True)
        cls.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False)
        cls.post_pred = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        cls.post_label = AsDiscrete(threshold=0.5)

    def test_model_forward_pass(self):
        """Test that the model produces output of the correct shape."""
        # Create a dummy input (batch=1, channels=1, D=32, H=32, W=32)
        # Using 32 as it's divisible by 16 (required by 4 downsampling layers)
        dummy_input = torch.randn(1, 1, 32, 32, 32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)

        # Output should have 2 channels (background + heart)
        expected_shape = (1, 2, 32, 32, 32)
        self.assertEqual(output.shape, expected_shape,
                         f"Expected output shape {expected_shape}, got {output.shape}")

    def test_loss_computation(self):
        """Test that loss can be computed without errors."""
        dummy_input = torch.randn(1, 1, 32, 32, 32).to(self.device)
        dummy_label = torch.randint(
            0, 2, (1, 1, 32, 32, 32)).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)
            loss = self.loss_function(output, dummy_label)

        self.assertIsInstance(loss.item(), float,
                              "Loss should be a float value")
        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss should not be infinite")

    def test_dice_metric_computation(self):
        """Test that DiceMetric can be computed with post-processing."""
        dummy_input = torch.randn(1, 1, 32, 32, 32).to(self.device)
        dummy_label = torch.randint(
            0, 2, (1, 1, 32, 32, 32)).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)
            output_post = self.post_pred(output)
            label_post = self.post_label(dummy_label)
            self.dice_metric(y_pred=output_post, y=label_post)
            metric = self.dice_metric.aggregate().item()
            self.dice_metric.reset()

        self.assertIsInstance(
            metric, float, "Dice metric should be a float value")
        self.assertGreaterEqual(metric, 0.0, "Dice metric should be >= 0")
        self.assertLessEqual(metric, 1.0, "Dice metric should be <= 1")


class TestTrainingStep(unittest.TestCase):
    """Test a single training step."""

    def test_training_step(self):
        """Test that a single training step works correctly."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)

        loss_function = DiceLoss(
            to_onehot_y=True, sigmoid=True, squared_pred=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        dummy_input = torch.randn(1, 1, 32, 32, 32).to(device)
        dummy_label = torch.randint(
            0, 2, (1, 1, 32, 32, 32)).float().to(device)

        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = loss_function(output, dummy_label)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(
            loss), "Loss should not be NaN after training step")


if __name__ == "__main__":
    unittest.main()
